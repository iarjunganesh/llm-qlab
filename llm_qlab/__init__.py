"""llm-qlab — measurement logic for quantized LLM inference benchmarking.

This package holds the two importable modules. The CLI entry points stay at the
repository root, because every command in the README invokes them directly
(``python benchmark.py ...``) and moving them would break the documented
interface for no gain.

The split that matters is not library-versus-script, it is
llama_cpp-versus-no-llama_cpp:

* :mod:`llm_qlab.bench_core` imports ``llama_cpp`` and needs a CUDA build.
* :mod:`llm_qlab.results_schema` deliberately does not, so ``compare_quants.py``
  runs anywhere — you do not need a GPU to plot results someone else measured.

Keep that property when adding to this package.
"""
