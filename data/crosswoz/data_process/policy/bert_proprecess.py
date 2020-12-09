from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

MAX_SEQ_LEN = 512


class PolicyDataset(Dataset):
    def __init__(
        self,
        dial_ids: Optional[List[str]] = None,
        turn_ids: Optional[List[str]] = None,
        input_ids: Optional[List[List[int]]] = None,
        token_type_ids: Optional[List[List[int]]] = None,
        labels: Optional[List[List[int]]] = None,
    ):
        super(PolicyDataset, self).__init__()
        self.dial_ids = dial_ids
        self.turn_ids = turn_ids
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.labels = labels

    def __getitem__(self, index: int) -> tuple:
        return (
            self.dial_ids[index],
            self.turn_ids[index],
            self.input_ids[index],
            self.token_type_ids[index],
            self.labels[index],
        )

    def __len__(self) -> int:
        return len(self.dial_ids)


def collate_fn(examples: List[tuple], mode: str = "train") -> Dict[str, torch.Tensor]:
    dial_ids, turn_ids, input_ids, token_type_ids, labels = list(zip(*examples))

    attention_mask, input_ids_tensor, token_type_ids_tensor = pad(
        input_ids, token_type_ids
    )

    data = {
        "input_ids": input_ids_tensor,
        "token_type_ids": token_type_ids_tensor,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.float),
    }

    if mode != "train":
        data.update(
            {
                "dial_ids": dial_ids,
                "turn_ids": turn_ids,
            }
        )

    return data


def pad(
    input_ids: List[List[int]], token_type_ids: List[List[int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad, truncate and generate attention mask.

    Args:
        input_ids: a list of input_ids
        token_type_ids: a list of token_type_ids
    Returns:
        input_ids, token_type_ids, attention_mask tensor
    """
    max_seq_len = min(max(len(input_id) for input_id in input_ids), MAX_SEQ_LEN)
    # init
    input_ids_tensor = torch.zeros((len(input_ids), max_seq_len), dtype=torch.long)
    token_type_ids_tensor = torch.zeros_like(input_ids_tensor)
    attention_mask = torch.ones_like(input_ids_tensor)

    # pad and truncate
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


def str2id(
    tokenizer: BertTokenizer, sys_utter: str, usr_utter: str, source: str
) -> Tuple[List[int], List[int]]:
    """Convert system, user utterance and source tokens to ids based on BertTokenizer.

    Args:
        tokenizer: BertTokenizer
        sys_utter: system utterance
        usr_utter: user utterance
        source: slot + value

    Returns:
        input_ids and token_type_ids
    """
    sys_utter_tokens = tokenizer.tokenize(sys_utter)
    usr_utter_tokens = tokenizer.tokenize(usr_utter)
    source_tokens = tokenizer.tokenize(source)
    sys_utter_ids = tokenizer.convert_tokens_to_ids(sys_utter_tokens)
    usr_utter_ids = tokenizer.convert_tokens_to_ids(usr_utter_tokens)
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    input_ids = (
        [tokenizer.cls_token_id]
        + sys_utter_ids
        + [tokenizer.sep_token_id]
        + usr_utter_ids
        + [tokenizer.sep_token_id]
        + source_ids
        + [tokenizer.sep_token_id]
    )
    token_type_ids = (
        [0]
        + [0] * (len(sys_utter_ids) + 1)
        + [1] * (len(usr_utter_ids) + 1)
        + [0] * (len(source_ids) + 1)
    )
    return input_ids, token_type_ids
