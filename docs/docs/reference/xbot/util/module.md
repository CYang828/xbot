---
sidebar_label: module
title: xbot.util.module
---

module interface.

## Module Objects

```python
class Module(ABC)
```

#### train

```python
 | train(*args, **kwargs)
```

Model training entry point

#### test

```python
 | test(*args, **kwargs)
```

Model testing entry point

#### from\_cache

```python
 | from_cache(*args, **kwargs)
```

restore internal state for multi-turn dialog

#### to\_cache

```python
 | to_cache(*args, **kwargs)
```

save internal state for multi-turn dialog

#### init\_session

```python
 | init_session()
```

Init the class variables for a new session.

