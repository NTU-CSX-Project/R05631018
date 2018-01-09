## Chat-Bot

This is a Chinese chat-bot based on seq2seq model.

The datasets are from 民視八點檔

The code will be released after 2018/1/15. Here is a api to demo the chatbot.

```python
chatbot('我是真心想向妳保證')
```

It may return "但我現在已經知道了"


### Some Examples:

```python
chatbot('我也知道')
```
- Output:

    '但我現在已經知道'
    
```python
chatbot('安安你好給虧嗎')
```
- Output:

    '我是真心想向妳道謝'
    
↑ "給虧" is not in the vocabulary
