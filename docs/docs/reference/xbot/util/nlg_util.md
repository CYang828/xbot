---
sidebar_label: nlg_util
title: xbot.util.nlg_util
---

Natural Language Generation Interface

## NLG Objects

```python
class NLG(Module)
```

Base class for NLG model.

#### generate

```python
 | generate(action)
```

Generate a natural language utterance conditioned on the dialog act.

**Arguments**:

  action (list of list):
  The dialog action produced by dialog policy module, which is in dialog act format.

**Returns**:

  utterance (str):
  A natural langauge utterance.

