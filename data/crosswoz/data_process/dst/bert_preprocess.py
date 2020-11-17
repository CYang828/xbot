import torch
from torch.utils.data import Dataset

MAX_SEQ_LEN = 512


def turn2examples(tokenizer, domain, slot, value, context_ids, triple_labels=None,
                  belief_state=None, dial_id=None, turn_id=None):
    candidate = domain + '-' + slot + ' = ' + value
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    input_ids = ([tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id]
                 + candidate_ids + [tokenizer.sep_token_id])
    token_type_ids = [0] + [0] * len(context_ids) + [0] + [1] * len(candidate_ids) + [1]
    example = (input_ids, token_type_ids, domain, slot, value)
    if dial_id is not None:
        label = int((domain, slot, value) in triple_labels)
        example += (belief_state, label, dial_id, str(turn_id))
    return example


class DSTDataset(Dataset):

    def __init__(self, examples):
        super(DSTDataset, self).__init__()
        if len(examples) > 5:
            (self.input_ids, self.token_type_ids, self.domains, self.slots, self.values,
             self.belief_states, self.labels, self.dialogue_idxs, self.turn_ids) = examples
        else:
            (self.input_ids, self.token_type_ids, self.domains, self.slots, self.values) = examples

    def __getitem__(self, index):
        if hasattr(self, 'labels'):
            return (self.input_ids[index], self.token_type_ids[index], self.domains[index],
                    self.slots[index], self.values[index], self.belief_states[index],
                    self.labels[index], self.dialogue_idxs[index], self.turn_ids[index])
        else:
            return (self.input_ids[index], self.token_type_ids[index], self.domains[index],
                    self.slots[index], self.values[index])

    def __len__(self):
        return len(self.input_ids)


def collate_fn(examples, mode='train'):
    batch_examples = {}
    examples = list(zip(*examples))

    if mode == 'infer':
        batch_examples['domains'], batch_examples['slots'], batch_examples['values'] = examples[2:]
    else:
        (batch_examples['domains'], batch_examples['slots'], batch_examples['values'],
         batch_examples['belief_states'], batch_examples['labels'], batch_examples['dialogue_idxs'],
         batch_examples['turn_ids']) = examples[2:]

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
            token_type_ids_tensor[i, :cur_seq_len] = torch.tensor(token_type_ids[i], dtype=torch.long)
            attention_mask[i, cur_seq_len:] = 0
        else:
            input_ids_tensor[i] = torch.tensor(input_id[:max_seq_len - 1] + [102], dtype=torch.long)
            token_type_ids_tensor[i] = torch.tensor(token_type_ids[i][:max_seq_len], dtype=torch.long)

    data = {
        'input_ids': input_ids_tensor,
        'token_type_ids': token_type_ids_tensor,
        'attention_mask': attention_mask,
    }

    if mode == 'infer':
        data.update({
            'domains': batch_examples['domains'],
            'slots': batch_examples['slots'],
            'values': batch_examples['values']
        })
        return data
    else:
        data.update({
            'labels': torch.tensor(batch_examples['labels'], dtype=torch.long)
        })

    if mode != 'train':
        data.update({
            'domains': batch_examples['domains'],
            'slots': batch_examples['slots'],
            'values': batch_examples['values'],
            'belief_states': batch_examples['belief_states'],
            'dialogue_idxs': batch_examples['dialogue_idxs'],
            'turn_ids': batch_examples['turn_ids']
        })

    return data
