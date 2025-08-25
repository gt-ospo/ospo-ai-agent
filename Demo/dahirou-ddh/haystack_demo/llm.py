"""
Loads Llama-3-8B (GGUF) via llama-cpp-haystack
Usage: from llm import llm_gen
"""
import haystack.tools as ht

if not hasattr(ht, "_check_duplicate_tool_names"):
    def _check_duplicate_tool_names(tools):
        return tools
    ht._check_duplicate_tool_names = _check_duplicate_tool_names

if not hasattr(ht, "deserialize_tools_inplace"):
    ht.deserialize_tools_inplace = lambda tools: tools

from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
import config as C

_llm = LlamaCppGenerator(model=str(C.GGUF_PATH), n_ctx=C.N_CTX)
_llm.model_kwargs.update({
    "n_threads": C.N_THREADS,
    "n_gpu_layers": C.N_GPU_LAYERS
})
_llm.warm_up()

def llm_gen(prompt: str, **overrides):
    """
    Call this instead of _llm.run(...) to get consistent defaults:
      • temperature = 0.7
      • max_tokens  = C.MAX_TOKENS
    Any override in **overrides (e.g. top_p=0.9) will be applied.
    Returns a plain string (the model’s text).
    """
    gen_kwargs = {"temperature": 0.1, "max_tokens": C.MAX_TOKENS}
    gen_kwargs.update(overrides)
    out = _llm.run(prompt=prompt, generation_kwargs=gen_kwargs)
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict) and "replies" in out:
        first = out["replies"][0]
        return first.text if hasattr(first, "text") else str(first)
    return str(out)

print("Llama-3-8B loaded; call llm.llm_gen(prompt)")

