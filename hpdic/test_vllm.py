from vllm import LLM, SamplingParams
import gc  # <--- 新增引用

# 1. 准备提示词
prompts = [
    "Hello, my name is",
    "The capital of France is",
]

# 2. 设置采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 3. 初始化 vLLM 引擎
llm = LLM(model="facebook/opt-125m")

# 4. 生成结果
outputs = llm.generate(prompts, sampling_params)

# 5. 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# === 新增：优雅退出代码 ===
# 强制删除引擎对象，触发它的析构函数（__del__），让它有机会正常关闭子进程
del llm
# 强制进行垃圾回收，确保内存立即释放
gc.collect()
# ========================