---
sidebar_label: dst_util
title: xbot.util.dst_util
---

Dialog State Tracker Interface

## DST Objects

```python
class DST(Module)
```

Base class for dialog state tracker models.

#### update

```python
 | update(action)
```

Update the internal dialog state variable.
update state[&#x27;user_action&#x27;] with input action

**Arguments**:

  action (str or list of tuples):
  The type is str when DST is word-level (such as NBT), and list of tuples when it is DA-level.

**Returns**:

  new_state (dict):
  Updated dialog state, with the same form of previous state.

