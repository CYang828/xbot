from collections import defaultdict
from typing import List, Tuple, Set, Optional

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer

MAX_SEQ_LEN = 512


def turn2example(
    tokenizer: BertTokenizer,
    domain: str,
    slot: str,
    value: str,
    context_ids: List[int],
    triple_labels: Optional[Set[tuple]] = None,
    belief_state: Optional[List[Tuple[str, str, str]]] = None,
    dial_id: str = None,
    turn_id: int = None,
) -> tuple:
    """Convert turn data to example based on ontology.

    Args:
        tokenizer: BertTokenizer, see https://huggingface.co/transformers/model_doc/bert.html#berttokenizer
        domain: domain of current example
        slot: slot of current example
        value: value of current example
        context_ids: context token's id in bert vocab
        triple_labels: set of (domain, slot, value)
        belief_state: list of (domain, slot, value)
        dial_id: current dialogue id
        turn_id: current turn id

    Returns:
        example, (input_ids, token_type_ids, domain, slot, value, ...)
    """
    candidate = domain + "-" + slot + " = " + value
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    input_ids = (
        [tokenizer.cls_token_id]
        + context_ids
        + [tokenizer.sep_token_id]
        + candidate_ids
        + [tokenizer.sep_token_id]
    )
    token_type_ids = [0] + [0] * len(context_ids) + [0] + [1] * len(candidate_ids) + [1]
    example = (input_ids, token_type_ids, domain, slot, value)
    if dial_id is not None:
        label = int((domain, slot, value) in triple_labels)
        example += (belief_state, label, dial_id, str(turn_id))
    return example


class DSTDataset(Dataset):
    def __init__(self, examples: List[tuple]):
        super(DSTDataset, self).__init__()
        if len(examples) > 5:
            (
                self.input_ids,
                self.token_type_ids,
                self.domains,
                self.slots,
                self.values,
                self.belief_states,
                self.labels,
                self.dialogue_idxs,
                self.turn_ids,
            ) = examples
        else:
            (
                self.input_ids,
                self.token_type_ids,
                self.domains,
                self.slots,
                self.values,
            ) = examples

    def __getitem__(self, index: int) -> tuple:
        if hasattr(self, "labels"):
            return (
                self.input_ids[index],
                self.token_type_ids[index],
                self.domains[index],
                self.slots[index],
                self.values[index],
                self.belief_states[index],
                self.labels[index],
                self.dialogue_idxs[index],
                self.turn_ids[index],
            )
        else:
            return (
                self.input_ids[index],
                self.token_type_ids[index],
                self.domains[index],
                self.slots[index],
                self.values[index],
            )

    def __len__(self):
        return len(self.input_ids)


def collate_fn(examples: List[tuple], mode: str = "train") -> dict:
    """Merge a list of samples to form a mini-batch of Tensor(s)

    generate input_id tensor, token_type_id tensor, attention_mask tensor, pad all tensor to the longest
    sequence in the batch.

    Args:
        examples: list of (input_ids, token_type_ids, domain, slot, value, ...)
        mode: train, dev, tests, infer

    Returns:
        batch data
    """
    batch_examples = {}
    examples = list(zip(*examples))

    if mode == "infer":
        (
            batch_examples["domains"],
            batch_examples["slots"],
            batch_examples["values"],
        ) = examples[2:]
    else:
        (
            batch_examples["domains"],
            batch_examples["slots"],
            batch_examples["values"],
            batch_examples["belief_states"],
            batch_examples["labels"],
            batch_examples["dialogue_idxs"],
            batch_examples["turn_ids"],
        ) = examples[2:]

    attention_mask, input_ids_tensor, token_type_ids_tensor = get_bert_input(examples)

    data = {
        "input_ids": input_ids_tensor,
        "token_type_ids": token_type_ids_tensor,
        "attention_mask": attention_mask,
    }

    if mode == "infer":
        data.update(
            {
                "domains": batch_examples["domains"],
                "slots": batch_examples["slots"],
                "values": batch_examples["values"],
            }
        )
        return data
    else:
        data.update(
            {"labels": torch.tensor(batch_examples["labels"], dtype=torch.long)}
        )

    if mode != "train":
        data.update(
            {
                "domains": batch_examples["domains"],
                "slots": batch_examples["slots"],
                "values": batch_examples["values"],
                "belief_states": batch_examples["belief_states"],
                "dialogue_idxs": batch_examples["dialogue_idxs"],
                "turn_ids": batch_examples["turn_ids"],
            }
        )

    return data


def get_bert_input(
    examples: List[tuple],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input list to torch tensor.

    Args:
        examples: (input_id_list, )

    Returns:
        attention_mask, input_ids_tensor, token_type_ids_tensor
    """
    input_ids = examples[0]
    token_type_ids = examples[1]
    max_seq_len = min(max(len(input_id) for input_id in input_ids), MAX_SEQ_LEN)
    input_ids_tensor = torch.zeros((len(input_ids), max_seq_len), dtype=torch.long)
    token_type_ids_tensor = torch.zeros_like(input_ids_tensor)
    attention_mask = torch.ones_like(input_ids_tensor)

    for i, input_id in enumerate(input_ids):
        cur_seq_len = len(input_id)
        if cur_seq_len <= max_seq_len:
            input_ids_tensor[i, :cur_seq_len] = torch.tensor(input_id, dtype=torch.long)
            token_type_ids_tensor[i, :cur_seq_len] = torch.tensor(
                token_type_ids[i], dtype=torch.long
            )
            attention_mask[i, cur_seq_len:] = 0
        else:
            input_ids_tensor[i] = torch.tensor(
                input_id[: max_seq_len - 1] + [102], dtype=torch.long
            )
            token_type_ids_tensor[i] = torch.tensor(
                token_type_ids[i][:max_seq_len], dtype=torch.long
            )

    return attention_mask, input_ids_tensor, token_type_ids_tensor


def rank_values(logits: List[float], preds: List[tuple], top_k: int) -> List[tuple]:
    """Rank domain-slot pair corresponding values.

    Args:
        logits: prediction corresponding logits
        preds: a list of triple labels
        top_k: take top k prediction labels

    Returns:
        top-1 predicted triple label
    """
    top_k = min(len(preds), top_k)
    preds_logits_pair = sorted(zip(preds, logits), key=lambda x: x[1], reverse=True)[
        :top_k
    ]
    ranking_dict = defaultdict(list)
    for pred, logit in preds_logits_pair:
        key = "-".join(pred[:2])
        ranking_dict[key].append((pred[-1], logit))
    preds = []
    for ds, pred_score in ranking_dict.items():
        domain, slot = ds.split("-")
        value = sorted(pred_score, key=lambda x: x[-1], reverse=True)[0][0]
        preds.append((domain, slot, value))
    return preds
