import ctypes

import llama_cpp

def token_to_string(model: llama_cpp.llama_model_p, token_id: llama_cpp.llama_token):
    try:
        buf = ctypes.c_char_p(b'\0'*16)
        size = llama_cpp.llama_token_to_piece(model, token_id, buf, 16, 0, False)
        if size >= 0:
            return buf.value[:size].decode('utf-8')
        size = -size
        buf = ctypes.c_char_p(b'\0'*(size))
        size = llama_cpp.llama_token_to_piece(model, token_id, buf, size, 0, False)
        return buf.value[:size].decode('utf-8')
    except UnicodeDecodeError:
        return "?"
