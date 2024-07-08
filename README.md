# cog-comfyui-liveportrait

> Efficient Portrait Animation with Stitching and Retargeting Control

A cog implementation of LivePortrait using the ComfyUI custom node

- Paper: https://arxiv.org/pdf/2407.03168
- Website: https://liveportrait.github.io/

## Example driving videos

Try these videos:

https://github.com/KwaiVGI/LivePortrait/tree/main/assets/examples/driving

## License

LivePortrait uses InsightFace `buffalo_l` models, meaning this model cannot be used commercially. The LivePortrait code and weights are released under an [MIT license](https://github.com/KwaiVGI/LivePortrait?tab=MIT-1-ov-file#readme).

## Implementation

This model uses the ComfyUI custom node created by Kijai:

https://github.com/kijai/ComfyUI-LivePortraitKJ

And the safetensor weights they converted:

https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main
