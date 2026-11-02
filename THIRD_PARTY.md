# Source and component notices

The final submission's original notices are preserved:

| Component | Notice |
| --- | --- |
| MetaFine source components | `LICENSE`, `vendor/METAFINE_DINO_DIT_LICENSE` |
| T1 release | `t1_best/LICENSE` |
| OpenPI / openpi_client | `vendor/OPENPI_LICENSE` |
| DINOv2 | `vendor/DINOV2_LICENSE` |
| Diffusion Policy components | `vendor/DIFFUSION_POLICY_LICENSE`, `policy/t5_combo/t5_target_dp/vendor/LICENSE` |

The OpenPI subtree is the inference dependency subset from the scored submission.
The T5 subtree includes the runtime/model definitions required by the final
seven-component combination. Removing unselected submission backends does not
change the retained third-party licenses.

The model bundle is distributed separately, with its own component notices and
upstream references. This code's license does not assign a new blanket license
to pretrained model weights or the PaliGemma tokenizer.

Upstream projects:

- https://github.com/Physical-Intelligence/openpi
- https://github.com/facebookresearch/dinov2
- https://github.com/real-stanford/diffusion_policy
- https://github.com/Hiangx-robotics/MetaFine-Challenge_CoRL
