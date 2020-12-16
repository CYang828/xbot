---
sidebar_label: trade
title: xbot.dm.dst.trade_dst.trade
---

## Generator Objects

```python
class Generator(nn.Module)
```

#### attend

```python
 | @staticmethod
 | attend(seq, cond, lens)
```

attend over the sequences `seq` using the condition `cond`.

**Arguments**:

- `seq`: size (bs, seq_len, hidden)
- `cond`: size (bs, hidden)
- `lens`: 

**Returns**:



#### attend\_vocab

```python
 | @staticmethod
 | attend_vocab(seq, cond)
```

**Arguments**:

- `seq`: size (vocab_size, hidden)
- `cond`: size , (bs, hidden)

**Returns**:



