```bash
## Installation Instructions
source ~/ElasticIVF/myenv/bin/activate
cd ~/vllm
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="7.5"
export VLLM_INSTALL_PUNICA_KERNELS=0
pip install -e .  # took me about an hour...
python -c "import vllm; print('vLLM Version:', vllm.__version__); import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0))"

## Smoke Test
cd hpdic
export VLLM_ATTENTION_BACKEND=FLASHINFER
python test_vllm.py
```