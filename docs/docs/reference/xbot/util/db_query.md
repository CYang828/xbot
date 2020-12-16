---
sidebar_label: db_query
title: xbot.util.db_query
---

#### contains

```python
contains(arr, s)
```

反向逻辑，当 arr 中没有一个包含 s 的时候才为 true

## Database Objects

```python
class Database()
```

docstring for Database

#### query

```python
 | query(belief_state, cur_domain)
```

query database using belief state, return list of entities, same format as database

**Arguments**:

- `belief_state`: state[&#x27;belief_state&#x27;]
- `cur_domain`: maintain by DST, current query domain

**Returns**:

list of entities

