---
sidebar_label: nlu_util
title: xbot.util.nlu_util
---

Natural language understanding interface.

## NLU Objects

```python
class NLU(Module)
```

NLU module interface.

#### predict

```python
 | predict(utterance, context=list())
```

Predict the dialog act of a natural language utterance.

**Arguments**:

  utterance (string):
  A natural language utterance.
  context (list of string):
  Previous utterances.
  

**Returns**:

  action (list of list):
  The dialog act of utterance.

