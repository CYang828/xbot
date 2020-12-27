---
sidebar_label: bert
title: xbot.dm.policy.bert_policy.bert
---

## BertPolicy Objects

```python
class BertPolicy(Policy)
```

#### download\_data

```python
 | @staticmethod
 | download_data(infer_config: dict, model_dir: str) -> None
```

Download trained model for inference.

**Arguments**:

- `infer_config` - config used for inference
- `model_dir` - model save directory

#### load\_config

```python
 | @staticmethod
 | load_config() -> dict
```

Load config for inference.

**Returns**:

  config dict

#### preprocess

```python
 | preprocess(belief_state: Dict[str, dict], cur_domain: str, history: List[tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Preprocess raw dialogue data to bert inputs.

**Arguments**:

- `belief_state` - see `src/xbot/util/state.py`
- `cur_domain` - current domain
- `history` - dialogue history, [(&#x27;usr&#x27;, &#x27;xxx&#x27;), (&#x27;sys&#x27;, &#x27;xxx&#x27;), ...]
  

**Returns**:

  bert inputs, contain input_ids, token_type_ids, attention_mask

#### get\_source

```python
 | @staticmethod
 | get_source(belief_state: Dict[str, dict], cur_domain: str) -> str
```

Take constraints in belief state.
TODO: belief_state 要不要换成 sys_state

**Arguments**:

- `belief_state` - current belief state
- `cur_domain` - current domain
  

**Returns**:

  concatenate all slot-value pair

#### predict

```python
 | predict(state: dict) -> List[list]
```

Predict the next actions of system.

**Arguments**:

- `state` - current system state
  

**Returns**:

  a list of actions of system will take

#### forward

```python
 | forward(belief_state: Dict[str, dict], cur_domain: str, history: List[tuple]) -> torch.Tensor
```

Forward step, get predictions.

**Arguments**:

- `belief_state` - see `src/xbot/util/state.py`
- `cur_domain` - current domain
- `history` - dialogue history, [(&#x27;usr&#x27;, &#x27;xxx&#x27;), (&#x27;sys&#x27;, &#x27;xxx&#x27;), ...]
  

**Returns**:

  model predictions

#### get\_sys\_das

```python
 | get_sys_das(db_res: dict, domain: str, intent: str, slot: str, sys_das: list) -> None
```

Construct system actions according to different domains and values taken from database.

**Arguments**:

- `db_res` - database query results
- `domain` - current domain
- `intent` - system&#x27;s intent, such as: Inform, Recommend,...
- `slot` - candidate slot, such ad: &#x27;价格&#x27;, &#x27;名称&#x27;, ...
- `sys_das` - system actions are saved into sys_das

#### get\_metro\_das

```python
 | @staticmethod
 | get_metro_das(db_res: dict, slot: str) -> str
```

Take departure and destination of metro from database query results.

**Arguments**:

- `db_res` - database query results
- `slot` - &#x27;起点&#x27; or &#x27;终点&#x27;
  

**Returns**:

  specified departure or destination

