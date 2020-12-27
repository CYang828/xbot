---
sidebar_label: policy_util
title: xbot.util.policy_util
---

Policy Interface

## Policy Objects

```python
class Policy(Module)
```

Base class for policy model.

#### predict

```python
 | predict(state)
```

Predict the next agent action given dialog state.
update state[&#x27;system_action&#x27;] with predict system action

**Arguments**:

  state (tuple or dict):
  when the DST and Policy module are separated, the type of state is tuple.
  else when they are aggregated together, the type of state is dict (dialog act).

**Returns**:

  action (list of list):
  The next dialog action.

