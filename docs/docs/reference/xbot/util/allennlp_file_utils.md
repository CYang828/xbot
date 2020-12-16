---
sidebar_label: allennlp_file_utils
title: xbot.util.allennlp_file_utils
---

Copy from allennlp https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py
Utilities for working with the local dataset cache.

## Tqdm Objects

```python
class Tqdm()
```

#### set\_slower\_interval

```python
 | @staticmethod
 | set_slower_interval(use_slower_interval: bool) -> None
```

If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm&#x27;s`` default
output rate.  ``tqdm&#x27;s`` default output rate is great for interactively watching progress,
but it is not great for log files.  You might want to set this if you are primarily going
to be looking at output through log files, not the terminal.

#### url\_to\_filename

```python
url_to_filename(url: str, etag: str = None) -> str
```

Convert `url` into a hashed filename in a repeatable way.
If `etag` is specified, append its hash to the url&#x27;s, delimited
by a period.

#### filename\_to\_url

```python
filename_to_url(filename: str, cache_dir: str = None) -> Tuple[str, str]
```

Return the url and etag (which may be ``None``) stored for `filename`.
Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.

#### cached\_path

```python
cached_path(url_or_filename: Union[str, Path], cache_dir: str = None) -> str
```

Given something that might be a URL (or might be a local path),
determine which. If it&#x27;s a URL, download the file and cache it, and
return the path to the cached file. If it&#x27;s already a local path,
make sure the file exists and then return the path.

#### is\_url\_or\_existing\_file

```python
is_url_or_existing_file(url_or_filename: Union[str, Path, None]) -> bool
```

Given something that might be a URL (or might be a local path),
determine check if it&#x27;s url or an existing file path.

#### split\_s3\_path

```python
split_s3_path(url: str) -> Tuple[str, str]
```

Split a full s3 path into the bucket name and path.

#### s3\_request

```python
s3_request(func: Callable)
```

Wrapper function for s3 requests in order to create more helpful error
messages.

#### s3\_etag

```python
@s3_request
s3_etag(url: str) -> Optional[str]
```

Check ETag on S3 object.

#### s3\_get

```python
@s3_request
s3_get(url: str, temp_file: IO) -> None
```

Pull a file directly from S3.

#### session\_with\_backoff

```python
session_with_backoff() -> requests.Session
```

We ran into an issue where http requests to s3 were timing out,
possibly because we were making too many requests too quickly.
This helper function returns a requests session that has retry-with-backoff
built in.

see https://stackoverflow.com/questions/23267409/how-to-implement-retry-mechanism-into-python-requests-library

#### get\_from\_cache

```python
get_from_cache(url: str, cache_dir: str = None) -> str
```

Given a URL, look for the corresponding dataset in the local cache.
If it&#x27;s not there, download it. Then return the path to the cached file.

#### read\_set\_from\_file

```python
read_set_from_file(filename: str) -> Set[str]
```

Extract a de-duped collection (set) of text from a file.
Expected file format is one item per line.

