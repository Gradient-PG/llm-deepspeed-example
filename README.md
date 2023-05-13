# LLM DeepSpeed Example
Simple PoC to train CLM models from HuggingFace with DeepSpeed through the trainer integration.
It'll work with any `csv` file with a `text` column.
For configs refer to the [integration](https://huggingface.co/docs/transformers/main_classes/deepspeed#deployment-with-multiple-gpus) or [DeepSpeed](https://www.deepspeed.ai/getting-started/) docs.

## Setup
```
pip install -r requirements.txt
```

## Training
As simple as:
```
deepspeed train_clm.py --deepspeed <path_to_config> --data_path <path_to_csv>
```
Note that the arguments for the training script need to come after `train_clm.py`, otherwise they'd be treated as DeepSpeed arguments.