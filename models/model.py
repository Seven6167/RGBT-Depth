"""Unified fixed-depth and instance-adaptive inference model."""

import ast
from collections import Counter
import inspect
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FusionPro


VALID_DEPTHS = (1, 2, 3, 4)


INACTIVE_BLOCK_ATTRIBUTES = ("mamba", "attention", "frequency")


class FusionDepthController(nn.Module):
    """Four-depth controller using pre-fusion modality features."""

    def __init__(self, feature_channels=128, hidden_channels=128, num_depths=4):
        super().__init__()
        input_channels = int(feature_channels) * 3
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.LayerNorm(input_channels),
            nn.Linear(input_channels, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), int(num_depths)),
        )

    def forward(self, rgb_feature, ir_feature):
        if rgb_feature.shape != ir_feature.shape:
            raise RuntimeError(
                f"Controller RGB/T shapes differ: {rgb_feature.shape} vs {ir_feature.shape}"
            )
        rgb = self.pool(rgb_feature).flatten(1)
        ir = self.pool(ir_feature).flatten(1)
        disagreement = self.pool((rgb_feature - ir_feature).abs()).flatten(1)
        return self.net(torch.cat([rgb, ir, disagreement], dim=1))


class MultiDepthFusionNetwork(nn.Module):
    """Shared fusion network with fixed and instance-adaptive inference depths."""

    def __init__(self, base_model, router_hidden=128):
        super().__init__()
        self._validate_base(base_model)
        self.base_model = base_model
        self._parameter_pruning = self._strip_inactive_teacher_only_parameters()
        feature_channels = int(base_model.fusion.conv_y_1.in_channels)
        self.router = FusionDepthController(feature_channels, router_hidden, len(VALID_DEPTHS))

        # The legacy sigmoid mask router is not used by Scope B.  It was loaded
        # strictly before wrapping, then removed so Student parameter counts and
        # checkpoints do not include an inactive router.
        if hasattr(self.base_model, "shared_router"):
            self.base_model.shared_router = nn.Identity()
        self.base_model.luyou_en = False
        self.assert_deployment_parameter_boundary()

    @staticmethod
    def _forward_self_attributes(module):
        """Return direct ``self.<name>`` reads in a module's forward method."""
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(type(module).forward)))
        except (OSError, TypeError, IndentationError, SyntaxError) as error:
            raise RuntimeError(
                f"Cannot audit {type(module).__name__}.forward before parameter pruning"
            ) from error
        return {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        }

    def _strip_inactive_teacher_only_parameters(self):
        """Remove inactive Block parameters as in the frozen shared-network model.

        The checkpoint must be loaded strictly into the resulting network.
        Inactive submodules are verified absent from Block.forward by AST.
        """
        removed_modules = []
        removed_parameters = 0
        for module_name, module in tuple(self.base_model.named_modules()):
            if type(module).__name__ != "Block":
                continue
            forward_attributes = self._forward_self_attributes(module)
            active = sorted(set(INACTIVE_BLOCK_ATTRIBUTES) & forward_attributes)
            if active:
                raise RuntimeError(
                    f"Refusing to prune active Block branches at {module_name}: {active}"
                )
            for attribute in INACTIVE_BLOCK_ATTRIBUTES:
                child = getattr(module, attribute, None)
                if child is None:
                    continue
                removed_parameters += sum(parameter.numel() for parameter in child.parameters())
                removed_modules.append(f"{module_name}.{attribute}")
                delattr(module, attribute)
        if not removed_modules or removed_parameters <= 0:
            raise RuntimeError(
                "Expected inactive Block parameters were not found; "
                "refusing an un-audited Student architecture"
            )
        return {
            "removed_module_count": len(removed_modules),
            "removed_parameter_count": int(removed_parameters),
            "removed_modules": tuple(removed_modules),
        }

    def parameter_boundary_report(self):
        state_keys = tuple(self.state_dict())
        named_parameters = tuple(name for name, _ in self.named_parameters())
        mssm_modules = tuple(
            name for name, module in self.named_modules()
            if type(module).__name__.lower() == "mssm"
        )
        mssm_parameters = tuple(
            name for name in named_parameters if "mssm" in name.lower()
        )
        legacy_router_parameters = tuple(
            name for name in named_parameters if "shared_router" in name
        )
        inactive_state_keys = tuple(
            key for key in state_keys
            if any(f".{attribute}." in key for attribute in INACTIVE_BLOCK_ATTRIBUTES)
        )
        return {
            "total_parameters": int(sum(p.numel() for p in self.parameters())),
            "base_parameters": int(sum(p.numel() for p in self.base_model.parameters())),
            "prefix_router_parameters": int(sum(p.numel() for p in self.router.parameters())),
            "removed_inactive_parameters": self._parameter_pruning["removed_parameter_count"],
            "removed_inactive_modules": self._parameter_pruning["removed_module_count"],
            "mssm_modules": list(mssm_modules),
            "mssm_parameters": list(mssm_parameters),
            "legacy_router_parameters": list(legacy_router_parameters),
            "inactive_block_state_keys": list(inactive_state_keys),
        }

    def assert_deployment_parameter_boundary(self):
        report = self.parameter_boundary_report()
        forbidden = (
            report["mssm_modules"]
            + report["mssm_parameters"]
            + report["legacy_router_parameters"]
            + report["inactive_block_state_keys"]
        )
        if forbidden:
            raise RuntimeError(f"Student parameter boundary audit failed: {forbidden[:20]}")
        return report

    @staticmethod
    def _validate_base(model):
        if model.__class__.__name__ != "FusionPro":
            raise RuntimeError(f"Multi-depth network requires FusionPro, got {type(model).__name__}")
        if model.fusion_mode != "fuse":
            raise RuntimeError("Multi-depth network requires fusion_mode=fuse")
        if bool(model.luyou_en):
            raise RuntimeError("Full-depth base must use luyou_en=0")
        lengths = [
            len(model.fusion.layers_y),
            len(model.fusion.layers_ir),
            len(model.fusion.layers_fuse),
            len(model.fusion.layers_fusion),
        ]
        if lengths != [4, 4, 4, 4]:
            raise RuntimeError(f"Expected four aligned fusion blocks, got {lengths}")

    def set_router_trainable(self, enabled):
        for parameter in self.router.parameters():
            parameter.requires_grad_(bool(enabled))

    def encode_backbone(self, x):
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"Expected Bx6xHxW input, got {tuple(x.shape)}")
        return self.base_model.rgb_enc(x[:, :3]), self.base_model.ir_enc(x[:, 3:6])

    def _shallow_tokens(self, rgb_feature, ir_feature):
        fusion = self.base_model.fusion
        y1 = fusion.conv_y_1(rgb_feature)
        y2 = fusion.conv_y_2(y1)
        ir1 = fusion.conv_ir_1(ir_feature)
        ir2 = fusion.conv_ir_2(ir1)
        fused = torch.cat([y1, ir1], dim=1)
        x_size = (y2.shape[2], y2.shape[3])
        y_tokens = fusion.pos_drop(fusion.patch_embed(y2))
        ir_tokens = fusion.pos_drop(fusion.patch_embed(ir2))
        fuse_tokens = fusion.pos_drop(fusion.patch_embed(fused))
        return y2, ir2, y_tokens, ir_tokens, fuse_tokens, x_size

    def _finish_heads(
        self, y2, ir2, y_tokens, ir_tokens, fuse_tokens, x_size,
        output_height, output_width,
    ):
        fusion = self.base_model.fusion
        y3 = fusion.patch_unembed(fusion.norm_y(y_tokens), x_size) + y2
        ir3 = fusion.patch_unembed(fusion.norm_ir(ir_tokens), x_size) + ir2
        fuse2 = fusion.patch_unembed(fusion.norm_fuse(fuse_tokens), x_size)
        y_out = fusion.y_conv_last_2(fusion.y_conv_last_1(y3))
        ir_out = fusion.ir_conv_last_2(fusion.ir_conv_last_1(ir3))
        fuse_out = fusion.conv_last_2(fusion.conv_last_1(fuse2))
        mode = self.base_model.fusion_mode
        if mode == "rgb":
            out = y_out
        elif mode == "ir":
            out = ir_out
        elif mode == "avg":
            out = 0.5 * (y_out + ir_out)
        else:
            out = fuse_out
        return F.interpolate(
            out,
            size=(output_height // 8, output_width // 8),
            mode="bilinear",
            align_corners=False,
        )

    def forward_prefix_from_backbone(
        self, rgb_feature, ir_feature, k, output_height, output_width,
        return_trace=False,
    ):
        k = int(k)
        if k not in VALID_DEPTHS:
            raise ValueError(f"k must be one of {VALID_DEPTHS}, got {k}")
        fusion = self.base_model.fusion
        y2, ir2, y_tokens, ir_tokens, fuse_tokens, x_size = self._shallow_tokens(
            rgb_feature, ir_feature
        )
        executed = []
        for index in range(k):
            y_tokens = fusion.layers_y[index](y_tokens, x_size)
            ir_tokens = fusion.layers_ir[index](ir_tokens, x_size)
            y_feature = fusion.patch_unembed(fusion.norm_y(y_tokens), x_size)
            ir_feature_stage = fusion.patch_unembed(fusion.norm_ir(ir_tokens), x_size)
            fuse_tokens = fusion.layers_fuse[index](fuse_tokens, x_size)
            fuse_feature = fusion.patch_unembed(fusion.norm_fuse(fuse_tokens), x_size)
            fuse_feature = fusion.layers_fusion[index](
                y_feature, ir_feature_stage, fuse_feature
            )
            fuse_tokens = fusion.pos_drop(fusion.patch_embed(fuse_feature))
            executed.append(index + 1)
        output = self._finish_heads(
            y2, ir2, y_tokens, ir_tokens, fuse_tokens, x_size,
            output_height, output_width,
        )
        if return_trace:
            return output, {"k": k, "executed_complete_stages": executed}
        return output

    def forward_fixed_k(self, x, k, return_trace=False):
        height, width = x.shape[-2:]
        rgb_feature, ir_feature = self.encode_backbone(x)
        return self.forward_prefix_from_backbone(
            rgb_feature, ir_feature, k, height, width, return_trace=return_trace
        )

    def route_logits(self, x):
        rgb_feature, ir_feature = self.encode_backbone(x)
        return self.router(rgb_feature, ir_feature)

    def forward_dynamic(self, x, return_aux=False):
        """True grouped execution. Batch=1 is the frozen latency protocol."""
        height, width = x.shape[-2:]
        rgb_feature, ir_feature = self.encode_backbone(x)
        logits = self.router(rgb_feature, ir_feature)
        selected_k = logits.argmax(dim=1) + 1
        outputs = x.new_empty((x.shape[0], 1, height // 8, width // 8))
        calls = Counter()
        for current_k in VALID_DEPTHS:
            indices = torch.nonzero(selected_k == current_k, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            subset = self.forward_prefix_from_backbone(
                rgb_feature.index_select(0, indices),
                ir_feature.index_select(0, indices),
                current_k,
                height,
                width,
            )
            outputs.index_copy_(0, indices, subset)
            calls[int(current_k)] += int(indices.numel())
        if return_aux:
            return outputs, {
                "route_logits": logits,
                "selected_k": selected_k,
                "selected_k_counts": dict(calls),
            }
        return outputs

    def forward(self, x, k=4, return_aux=False):
        if k == "dynamic" or k is None:
            return self.forward_dynamic(x, return_aux=return_aux)
        output = self.forward_fixed_k(x, int(k), return_trace=return_aux)
        return output


# Compatibility aliases for the previously verified RGBT-CC wrapper names.
PrefixRouter = FusionDepthController
ElasticFusionStudent = MultiDepthFusionNetwork
