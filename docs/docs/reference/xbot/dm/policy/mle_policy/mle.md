---
sidebar_label: mle
title: xbot.dm.policy.mle_policy.mle
---

## MultiDiscretePolicy Objects

```python
class MultiDiscretePolicy(nn.Module)
```

#### select\_action

```python
 | select_action(s, sample=True)
```

**Arguments**:

- `s`: [s_dim]
:param sample

**Returns**:

[a_dim]

#### get\_log\_prob

```python
 | get_log_prob(s, a)
```

**Arguments**:

- `s`: [b, s_dim]
- `a`: [b, a_dim]

**Returns**:

[b, 1]

