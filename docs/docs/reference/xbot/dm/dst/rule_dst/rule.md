---
sidebar_label: rule
title: xbot.dm.dst.rule_dst.rule
---

## RuleDST Objects

```python
class RuleDST(DST)
```

Rule based DST which trivially updates new values from NLU result to states.

#### init\_session

```python
 | init_session(state=None)
```

Initialize ``self.state`` with a default state.
:state: see xbot.util.state.default_state

#### update

```python
 | update(usr_da=None)
```

update belief_state, cur_domain, request_slot

**Arguments**:

- `usr_da`: List[List[intent, domain, slot, value]]

**Returns**:

state

