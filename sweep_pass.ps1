$ErrorActionPreference = "Continue"
$configs = @(
  @{f="llama2";  q="Q4_K_M"; m="models/llama-2-7b-chat.Q4_K_M.gguf"},
  @{f="llama2";  q="Q5_K_M"; m="models/llama-2-7b-chat.Q5_K_M.gguf"},
  @{f="llama2";  q="Q8_0";   m="models/llama-2-7b-chat.Q8_0.gguf"},
  @{f="mistral"; q="Q4_K_M"; m="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"},
  @{f="mistral"; q="Q5_K_M"; m="models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"},
  @{f="mistral"; q="Q8_0";   m="models/mistral-7b-instruct-v0.1.Q8_0.gguf"},
  @{f="qwen2.5"; q="Q4_K_M"; m="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"},
  @{f="qwen2.5"; q="Q5_K_M"; m="models/Qwen2.5-7B-Instruct-Q5_K_M.gguf"},
  @{f="qwen2.5"; q="Q8_0";   m="models/Qwen2.5-7B-Instruct-Q8_0.gguf"}
)
foreach ($c in $configs) {
  $log = "results/logs/C/C-$($c.f)-$($c.q).log"
  "=== $($c.f) $($c.q) === $(Get-Date -Format HH:mm:ss)"
  & .venv\Scripts\python.exe benchmark.py --model $c.m --quant-type $c.q --model-family $c.f --n-gpu-layers 99 --n-runs 5 *>&1 | Tee-Object -FilePath $log
}
"=== SWEEP COMPLETE $(Get-Date -Format HH:mm:ss) ==="
