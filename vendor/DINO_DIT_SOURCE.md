# T5 DINOv2-L + DiT

`policy/dino_dit/model.py` and `runtime.py` are copied from the evaluated
MetaFine T5 step-90000 implementation. Its MIT license is retained in
`METAFINE_DINO_DIT_LICENSE`. `backend.py` resolves assets within this submission
and loads the same full checkpoint with its EMA action parameters.

DINOv2-L originates from https://huggingface.co/facebook/dinov2-large and
https://github.com/facebookresearch/dinov2. Its Apache-2.0 license is retained
in `DINOV2_LICENSE`. The fine-tuned DINO weights are included in the checkpoint.

The optional U-Net implementation in `policy/dino_dit/vendor/` originates from
https://github.com/real-stanford/diffusion_policy; its MIT license is included
in that directory. T5 selects the DiT action backbone.
