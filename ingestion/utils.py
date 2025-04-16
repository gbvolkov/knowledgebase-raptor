# utils.py
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))
