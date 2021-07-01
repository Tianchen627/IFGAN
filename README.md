## Requirements
- `python3.6` 
- `PyTorch 1.3.0` 

## Prerequisite
Download preprocessed datasets from [google drive](https://drive.google.com/file/d/1zTss-wi7FGi3FmiqBZ8IFllgImoaC53_/view?usp=sharing),
and unzip it into `\data ` folder.The pretrained DistMult embeddings can be downloaded from [here](https://drive.google.com/file/d/1n1RYCxAEIOT713lFIBI41fuM9UKjNqXv/view),
Place these files in `\checkpoint\good_pretrain_init `
## Train the model
- Type command in `example.sh ` on terminal.
- Using different generator structure by filling args `--G_name` with different models from `\Model ` in `example.sh `.
- For example,`--G_name generator_concat-inv25` means using involution with embedding size 25.



## Reference
The code is inspired by [UPGAN](https://github.com/RichardHGL/UPGAN). 