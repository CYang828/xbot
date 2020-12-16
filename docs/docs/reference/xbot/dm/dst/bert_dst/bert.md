---
sidebar_label: bert
title: xbot.dm.dst.bert_dst.bert
---

## BertDST Objects

```python
class BertDST(DST)
```

#### download\_data

```python
 | @staticmethod
 | download_data(infer_config: dict) -> None
```

Download trained model and ontology file for inference.

**Arguments**:

- `infer_config` - config used for inference

#### load\_config

```python
 | @staticmethod
 | load_config() -> dict
```

Load config from common config and inference config from src/xbot/config/dst/bert .

**Returns**:

  config dict

#### preprocess

```python
 | preprocess(sys_uttr: str, usr_uttr: str) -> DataLoader
```

Preprocess raw utterance, convert them to token id for forward.

**Arguments**:

- `sys_uttr` - response of previous system turn
- `usr_uttr` - previous turn user&#x27;s utterance
  

**Returns**:

  DataLoader for inference

#### build\_examples

```python
 | build_examples(context_ids: List[int]) -> List[tuple]
```

Build examples according to ontology.

**Arguments**:

- `context_ids` - dialogue history id based on BertTokenizer
  

**Returns**:

  a list of example, (input_ids, token_type_ids, domain, slot, value)

#### init\_session

```python
 | init_session() -> None
```

Initiate state of one session.

#### update

```python
 | update(action: List[tuple]) -> None
```

Update session&#x27;s state according to output of bert.

**Arguments**:

- `action` - output of NLU module, but in bert dst, inputs are utterance of user and system,
  action is not used

#### update\_state

```python
 | update_state(pred_labels: List[tuple]) -> None
```

Update request slots and belief state in state.

**Arguments**:

- `pred_labels` - triple labels, (domain, slot, value)

#### forward

```python
 | forward(sys_uttr: str, usr_utter: str) -> List[tuple]
```

Bert model forward and rank output triple labels.

**Arguments**:

- `sys_uttr` - response of previous system turn
- `usr_utter` - previous turn user&#x27;s utterance
  

**Returns**:

  a list of triple labels, (domain, slot, value)

