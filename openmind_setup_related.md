# Installation

Installation related instructions can be found at the Megatron-LM fork.

## DeepSpeed

Clone the [repo](https://github.com/microsoft/DeepSpeed). Install using pip user.

# Training

```
export ROOT_DIR="/om2/user/chengxuz/megatron_related" ; bash examples/pretrain_gpt_distributed_1d3b.sh "${ROOT_DIR}/gpt_test_train/gpt2_1d3b_ds/ckpts" "${ROOT_DIR}/gpt_ckpts/gpt2-vocab.json" "${ROOT_DIR}/gpt_ckpts/gpt2-merges.txt" "/om2/group/evlab/llm_dataset/Megatron_datasets/pile/hf_dedp_data/pile_up_to_165-of-01650_text_document"
```
