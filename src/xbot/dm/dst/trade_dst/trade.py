import os
import sys
import json
import pickle

import numpy as np

import torch
import torch.nn as nn

from xbot.util.dst_util import DST
from xbot.util.state import default_state
from xbot.util.download import download_from_url
from xbot.util.path import get_data_path, get_config_path, get_root_path
from data.crosswoz.data_process.dst.trade_preprocess import (
    get_slot_information,
    prepare_data_for_update,
)

sys.path.append(os.path.join(get_root_path(), "script/dst/trade"))


class EncoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        dropout,
        n_layers,
        pad_id,
        pretrained_embedding_path="",
        load_embedding=False,
        fix_embedding=True,
    ):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True
        )

        if os.path.exists(pretrained_embedding_path) and load_embedding:
            with open(pretrained_embedding_path) as f:
                embedding = json.load(f)
            new = self.embedding.weight.data.new
            # new 生成一个和 self 相同设备和数据类型的 tensor，值为 E
            self.embedding.weight.data.copy_(new(embedding))
            self.embedding.weight.requires_grad = True

        if fix_embedding:
            self.embedding.weight.requires_grad = False

        print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

    def forward(self, input_seqs, input_lengths):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # seq_len, bs
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded)
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=False
            )
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(embedded)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = hidden[0] + hidden[1]  # bs, hidden, 用于解码
        outputs = (
            outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        )  # 双向不拼接，相加
        return outputs.transpose(0, 1), hidden.unsqueeze(
            0
        )  # (bs, seq_len, hidden), (1, bs, hidden)


class Generator(nn.Module):
    def __init__(
        self,
        lang,
        shared_emb,
        vocab_size,
        hidden_size,
        n_layer,
        dropout,
        slots,
        num_gates,
        parallel_decode,
    ):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layer, dropout=dropout)
        self.num_gates = num_gates
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3 * hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.parallel_decode = parallel_decode

        self.W_gate = nn.Linear(hidden_size, num_gates)

        # Create independent slot embeddings
        # 实际上是 domain 和 slot 的 w2i
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i:
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i:
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(
        self,
        batch_size,
        encoded_hidden,
        encoded_outputs,
        encoded_lens,
        story,
        max_res_len,
        target_batches,
        use_teacher_forcing,
        slot_temp,
    ):
        all_point_outputs = torch.zeros(
            len(slot_temp),
            batch_size,
            max_res_len,
            self.vocab_size,
            device=story.device,
        )
        all_gate_outputs = torch.zeros(
            len(slot_temp), batch_size, self.num_gates, device=story.device
        )

        # Get the slot embedding
        slot_emb_dict = {}
        # init
        slot_emb_arr = torch.zeros(
            (len(slot_temp), batch_size, self.hidden_size),
            dtype=torch.float,
            device=story.device,
        )
        for i, slot in enumerate(slot_temp):
            # init
            domain_emb = torch.zeros(
                (1, self.hidden_size), dtype=torch.float, device=story.device
            )
            slot_emb = torch.zeros(
                (1, self.hidden_size), dtype=torch.float, device=story.device
            )

            # Domain embedding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx, device=story.device)
                domain_emb = self.Slot_emb(domain_w2idx)

            # Slot embedding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx, device=story.device)
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one  不是拼接，直接相加, (hidden_size, )
            combined_emb = domain_emb + slot_emb
            slot_emb_dict[slot] = combined_emb
            # (1, bs, hidden_size)
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)
            # (num_slots, bs, hidden)
            slot_emb_arr[i, :, :] = slot_emb_exp

        if self.parallel_decode:
            # Compute pointer-generator output, putting all (domain, slot) in one batch
            decoder_input = self.dropout_layer(slot_emb_arr).view(
                -1, self.hidden_size
            )  # (batch*|slot|) * emb
            hidden = encoded_hidden.repeat(
                1, len(slot_temp), 1
            )  # 1 * (batch*|slot|) * emb

            self.gru.flatten_parameters()
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)

                enc_out = encoded_outputs.repeat(len(slot_temp), 1, 1)
                enc_len = encoded_lens * len(slot_temp)
                context_vec, logits, prob = self.attend(
                    enc_out, hidden.squeeze(0), enc_len
                )

                if wi == 0:
                    all_gate_outputs = torch.reshape(
                        self.W_gate(context_vec), all_gate_outputs.size()
                    )

                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat(
                    [dec_state.squeeze(0), context_vec, decoder_input], -1
                )
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_ptr = torch.zeros_like(p_vocab, device=story.device)
                p_context_ptr.scatter_add_(1, story.repeat(len(slot_temp), 1), prob)

                final_p_vocab = (1 - vocab_pointer_switches).expand_as(
                    p_context_ptr
                ) * p_context_ptr + vocab_pointer_switches.expand_as(
                    p_context_ptr
                ) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)

                all_point_outputs[:, :, wi, :] = torch.reshape(
                    final_p_vocab, (len(slot_temp), batch_size, self.vocab_size)
                )

                if use_teacher_forcing:
                    decoder_input = self.embedding(
                        torch.flatten(target_batches[:, :, wi].transpose(1, 0))
                    )
                else:
                    decoder_input = self.embedding(pred_word)

        else:
            # Compute pointer-generator output, decoding each (domain, slot) one-by-one
            words_point_out = []
            counter = 0
            self.gru.flatten_parameters()
            for slot in slot_temp:
                hidden = encoded_hidden  # (1, bs, hidden)
                words = []
                slot_emb = slot_emb_dict[slot]  # (hidden)
                decoder_input = self.dropout_layer(slot_emb).expand(
                    batch_size, self.hidden_size
                )
                for wi in range(max_res_len):
                    dec_state, hidden = self.gru(
                        decoder_input.expand_as(hidden), hidden
                    )
                    context_vec, logits, prob = self.attend(
                        encoded_outputs, hidden.squeeze(0), encoded_lens
                    )

                    if wi == 0:
                        all_gate_outputs[counter] = self.W_gate(context_vec)

                    p_vocab = self.attend_vocab(
                        self.embedding.weight, hidden.squeeze(0)
                    )  # (bs, vocab_size)
                    p_gen_vec = torch.cat(
                        [dec_state.squeeze(0), context_vec, decoder_input], -1
                    )
                    vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                    p_context_ptr = torch.zeros_like(
                        p_vocab, device=story.device
                    )  # (bs, vocab_size)
                    p_context_ptr.scatter_add_(
                        1, story, prob
                    )  # prob: (bs, seq_len), story: (bs, seq)

                    final_p_vocab = (1 - vocab_pointer_switches).expand_as(
                        p_context_ptr
                    ) * p_context_ptr + vocab_pointer_switches.expand_as(
                        p_context_ptr
                    ) * p_vocab
                    pred_word = torch.argmax(final_p_vocab, dim=1)  # (bs)

                    all_point_outputs[
                        counter, :, wi, :
                    ] = final_p_vocab  # (num_slots, bs, max_res_len, vocab_size)
                    if use_teacher_forcing:
                        decoder_input = self.embedding(
                            target_batches[:, counter, wi]
                        )  # Chosen word is next input
                    else:
                        decoder_input = self.embedding(pred_word)

                counter += 1
                words_point_out.append(words)
        # (num_slots, bs, max_res_len, vocab_size), (num_slots, bs, num_gates), (num_slots, max_res_len, bs)
        # 需要将 bs 转置到第一维度，否则 parallel gather 将出现问题
        all_point_outputs = all_point_outputs.transpose(0, 1)
        all_gate_outputs = all_gate_outputs.transpose(0, 1)
        return all_point_outputs, all_gate_outputs

    @staticmethod
    def attend(seq, cond, lens):
        """attend over the sequences `seq` using the condition `cond`.
        :param seq: size (bs, seq_len, hidden)
        :param cond: size (bs, hidden)
        :param lens:
        :return:
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores_.data[i, l:] = -np.inf
        scores = torch.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    @staticmethod
    def attend_vocab(seq, cond):
        """
        :param seq: size (vocab_size, hidden)
        :param cond: size , (bs, hidden)
        :return:
        """
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = torch.softmax(scores_, dim=1)
        return scores


class Trade(nn.Module):
    def __init__(
        self,
        lang,
        vocab_size,
        hidden_size,
        dropout,
        num_encoder_layers,
        num_decoder_layers,
        pad_id,
        slots,
        num_gates,
        unk_mask,
        pretrained_embedding_path="",
        load_embedding=False,
        fix_embedding=True,
        parallel_decode=False,
    ):
        super(Trade, self).__init__()
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.encoder = EncoderRNN(
            vocab_size,
            hidden_size,
            dropout,
            n_layers=num_encoder_layers,
            pad_id=pad_id,
            pretrained_embedding_path=pretrained_embedding_path,
            load_embedding=load_embedding,
            fix_embedding=fix_embedding,
        )
        self.decoder = Generator(
            lang,
            self.encoder.embedding,
            vocab_size,
            hidden_size,
            num_decoder_layers,
            dropout,
            slots,
            num_gates,
            parallel_decode,
        )

    def forward(self, data, use_teacher_forcing, slots):
        # Build unknown mask for memory to encourage generalization
        if self.unk_mask and self.decoder.training:
            story_size = data["context"].size()
            rand_mask = np.ones(story_size)
            # word dropout
            # >>> np.random.binomial([np.ones((3,4))], 0.5)
            # array([[[1, 1, 0, 1],
            #         [0, 0, 1, 1],
            #         [1, 1, 1, 0]]])
            bi_mask = np.random.binomial(
                [np.ones((story_size[0], story_size[1]))], 1 - self.dropout
            )[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.tensor(rand_mask, device=data["context"].device)
            story = data["context"] * rand_mask.long()
        else:
            story = data["context"]

        # Encode dialog history, output: (bs, seq_len, hidden), (1, bs, hidden)
        story = story[:, : data["context_len"].max()]
        encoded_outputs, encoded_hidden = self.encoder(
            story.transpose(0, 1), data["context_len"]
        )

        # Get the words that can be copy from the memory
        batch_size = len(data["context_len"])
        # (bs, num_slots, max_len)
        max_res_len = data["generate_y"].size(2) if self.encoder.training else 10
        all_point_outputs, all_gate_outputs = self.decoder(
            batch_size,
            encoded_hidden,
            encoded_outputs,
            data["context_len"],
            story,
            max_res_len,
            data["generate_y"],
            use_teacher_forcing,
            slots,
        )
        return all_point_outputs, all_gate_outputs


class TradeDST(DST):
    model_config_name = "dst/trade/inference.json"
    common_config_name = "dst/trade/common.json"

    model_urls = {
        "trade-Epoch-9-JACC-0.5060.pth": "http://qiw2jpwfc.hn-bkt.clouddn.com/trade-Epoch-9-JACC-0.5060.pth",
        "lang-train.pkl": "http://qiw2jpwfc.hn-bkt.clouddn.com/lang-train.pkl",
        "mem-lang-train.pkl": "http://qiw2jpwfc.hn-bkt.clouddn.com/mem-lang-train.pkl",
        "ontology.json": "http://qiw2jpwfc.hn-bkt.clouddn.com/ontology.json",
    }

    def __init__(self):
        super(TradeDST, self).__init__()
        # load config
        common_config_path = os.path.join(
            get_config_path(), TradeDST.common_config_name
        )
        common_config = json.load(open(common_config_path))
        model_config_path = os.path.join(get_config_path(), TradeDST.model_config_name)
        model_config = json.load(open(model_config_path))
        model_config.update(common_config)
        self.model_config = model_config
        self.model_config["data_path"] = os.path.join(
            get_data_path(), "crosswoz/dst_trade_data"
        )
        self.model_config["n_gpus"] = (
            0 if self.model_config["device"] == "cpu" else torch.cuda.device_count()
        )
        self.model_config["device"] = torch.device(self.model_config["device"])
        if model_config["load_embedding"]:
            model_config["hidden_size"] = 300

        # download data
        for model_key, url in TradeDST.model_urls.items():
            dst = os.path.join(self.model_config["data_path"], model_key)
            if model_key.endswith("pth"):
                file_name = "trained_model_path"
            elif model_key.endswith("pkl"):
                file_name = model_key.rsplit("-", maxsplit=1)[0]
            else:
                file_name = model_key.split(".")[0]  # ontology
            self.model_config[file_name] = dst
            if not os.path.exists(dst) or not self.model_config["use_cache"]:
                download_from_url(url, dst)

        # load date & model
        ontology = json.load(open(self.model_config["ontology"], "r", encoding="utf8"))
        self.all_slots = get_slot_information(ontology)
        self.gate2id = {"ptr": 0, "none": 1}
        self.id2gate = {id_: gate for gate, id_ in self.gate2id.items()}
        self.lang = pickle.load(open(self.model_config["lang"], "rb"))
        self.mem_lang = pickle.load(open(self.model_config["mem-lang"], "rb"))

        model = Trade(
            lang=self.lang,
            vocab_size=len(self.lang.index2word),
            hidden_size=self.model_config["hidden_size"],
            dropout=self.model_config["dropout"],
            num_encoder_layers=self.model_config["num_encoder_layers"],
            num_decoder_layers=self.model_config["num_decoder_layers"],
            pad_id=self.model_config["pad_id"],
            slots=self.all_slots,
            num_gates=len(self.gate2id),
            unk_mask=self.model_config["unk_mask"],
        )

        model.load_state_dict(torch.load(self.model_config["trained_model_path"]))

        self.model = model.to(self.model_config["device"]).eval()
        print(f'>>> {self.model_config["trained_model_path"]} loaded ...')
        self.state = default_state()
        print(">>> State initialized ...")

    def init_session(self):
        self.state = default_state()

    def update_belief_state(self, predict_belief):
        for domain_slot, value in predict_belief:
            domain, slot = domain_slot.split("-")
            if domain in self.state["belief_state"]:
                self.state["belief_state"][domain][slot] = value

    def update(self, user_action):
        source_text = " ; ".join(utter.strip() for _, utter in self.state["history"])
        curr_utterance = self.state["history"][-1][-1].strip()
        data_loader = prepare_data_for_update(
            self.model_config,
            self.lang,
            self.mem_lang,
            batch_size=1,
            source_text=source_text,
            curr_utterance=curr_utterance,
        )

        with torch.no_grad():
            data = next(iter(data_loader))
            predict_belief = []
            # (1, num_slots, resp_len, vocab_size), (1, num_slots, num_gates)
            all_ptr_words_probs, gates_logits = self.model(
                data, use_teacher_forcing=False, slots=self.all_slots
            )
            gate_ids = gates_logits[0].argmax(dim=-1)  # (num_slots, )

            all_ptr_word_ids = all_ptr_words_probs[0].argmax(
                dim=-1
            )  # (num_slots, resp_len)
            num_slots, resp_len = all_ptr_word_ids.size()
            for i in range(num_slots):
                slot_value = "none"

                if (
                    len(self.all_slots[i].split("-")) < 3
                    and self.id2gate[gate_ids[i].item()] == "ptr"
                ):
                    resp_tokens = []
                    for j in range(resp_len):
                        cur_token = self.lang.index2word[all_ptr_word_ids[i][j].item()]
                        if cur_token == "EOS":
                            break
                        resp_tokens.append(cur_token)
                    slot_value = " ".join(resp_tokens)

                if slot_value != "none":
                    predict_belief.append((self.all_slots[i], slot_value))

            self.update_belief_state(predict_belief)


if __name__ == "__main__":
    import random

    dst_model = TradeDST()
    data_path = os.path.join(get_data_path(), "crosswoz/dst_trade_data")
    dials_path = os.path.join(data_path, "dev_dials.json")
    # download dials file
    if not os.path.exists(dials_path):
        download_from_url(
            "http://qiw2jpwfc.hn-bkt.clouddn.com/dev_dials.json", dials_path
        )

    with open(os.path.join(data_path, "dev_dials.json"), "r", encoding="utf8") as f:
        dials = json.load(f)
        example = random.choice(dials)
        break_turn = 0
        for ti, turn in enumerate(example["dialogue"]):
            dst_model.state["history"].append(("sys", turn["system_transcript"]))
            dst_model.state["history"].append(("usr", turn["transcript"]))
            if random.random() < 0.5:
                break_turn = ti + 1
                break
    if break_turn == len(example["dialogue"]):
        print("对话已完成，请重新开始测试")
    print("对话状态更新前：")
    print(json.dumps(dst_model.state, indent=2, ensure_ascii=False))
    dst_model.update("")
    print("对话状态更新后：")
    print(json.dumps(dst_model.state, indent=2, ensure_ascii=False))
