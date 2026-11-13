"""SHA-pinned, weights-only checkpoint loading for both datasets."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import pickle
import pickletools
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Token:
    origin: int
    kind: str = 'opaque'
    literal: object = None
    identity: int = -1


class _Marker:
    def __init__(self, pos: int):
        self.pos = pos


def _prune_unneeded_root_values(data: bytes, required_keys: set[str]) -> bytes:
    ops = list(pickletools.genops(data))
    if not ops or ops[0][0].name != 'PROTO' or ops[0][1] != 2:
        raise RuntimeError('Expected torch.save protocol-2 pickle for metadata pruning')
    stack: list[_Token | _Marker] = []
    memo: dict[int, _Token] = {}
    patches: list[tuple[int,int,bytes]] = []
    root_identity = None
    next_identity = 1
    found: set[str] = set()

    def new(origin, kind='opaque', literal=None, ident=None):
        nonlocal next_identity
        if ident is None:
            ident=next_identity
            next_identity+=1
        return _Token(origin,kind,literal,ident)

    def pop_token():
        if not stack or not isinstance(stack[-1],_Token):
            raise RuntimeError('Malformed pickle token stack')
        return stack.pop()

    def last_marker():
        for index in range(len(stack)-1,-1,-1):
            if isinstance(stack[index],_Marker):
                return index
        raise RuntimeError('Malformed pickle MARK balance')

    def is_root(token):
        return isinstance(token,_Token) and token.identity==root_identity

    def record_pairs(pairs, end_pos):
        if len(pairs)%2:
            raise RuntimeError('Malformed top-level pickle dictionary pairs')
        for j in range(0,len(pairs),2):
            key,value=pairs[j:j+2]
            if not isinstance(key,_Token) or key.kind!='string' or not isinstance(key.literal,str):
                raise RuntimeError('Non-string top-level checkpoint metadata key')
            name=key.literal
            if name in found:
                raise RuntimeError(f'Duplicate checkpoint field: {name}')
            found.add(name)
            start=value.origin
            end=pairs[j+2].origin if j+2<len(pairs) else end_pos
            if not 0<=key.origin<start<=end<=len(data):
                raise RuntimeError('Unexpected non-contiguous top-level value expression')
            if name not in required_keys:
                patches.append((start,end,b'N'))

    atom_strings={'BINUNICODE','SHORT_BINUNICODE','UNICODE','BINSTRING','SHORT_BINSTRING','STRING'}
    atom_nums={'BININT','BININT1','BININT2','LONG','LONG1','LONG4','BINFLOAT','FLOAT'}
    for op,arg,pos in ops:
        name=op.name
        if name in {'PROTO','FRAME'}:
            continue
        if name == 'MARK':
            stack.append(_Marker(pos))
        elif name in atom_strings:
            stack.append(new(pos,'string' if isinstance(arg,str) else 'bytes',arg))
        elif name in atom_nums or name in {'NONE','NEWTRUE','NEWFALSE'}:
            stack.append(new(pos,'scalar',arg))
        elif name in {'EMPTY_DICT','EMPTY_LIST','EMPTY_TUPLE','EMPTY_SET'}:
            token = new(pos,'dict' if name=='EMPTY_DICT' else 'container')
            stack.append(token)
            if root_identity is None:
                if name!='EMPTY_DICT' or len(stack)!=1:
                    raise RuntimeError('Expected outer checkpoint dictionary')
                root_identity=token.identity
        elif name=='GLOBAL' or name=='STACK_GLOBAL':
            if name=='STACK_GLOBAL':
                pop_token();pop_token()
            stack.append(new(pos,'global',arg))
        elif name in {'BINPUT','LONG_BINPUT','PUT'}:
            if not stack or not isinstance(stack[-1],_Token):
                raise RuntimeError('Unexpected pickle memo put')
            memo[int(arg)]=stack[-1]
        elif name=='MEMOIZE':
            memo[len(memo)]=stack[-1]
        elif name in {'BINGET','LONG_BINGET','GET'}:
            if int(arg) not in memo:
                raise RuntimeError('Unknown pickle memo reference')
            previous=memo[int(arg)]
            stack.append(new(pos,previous.kind,previous.literal,previous.identity))
        elif name=='TUPLE1' or name=='TUPLE2' or name=='TUPLE3':
            count={'TUPLE1':1,'TUPLE2':2,'TUPLE3':3}[name]
            pieces=[pop_token() for _ in range(count)]
            stack.append(new(min(t.origin for t in pieces),'container'))
        elif name in {'TUPLE','LIST','DICT','FROZENSET'}:
            i=last_marker()
            marker=stack[i]
            del stack[i:]
            stack.append(new(marker.pos,'container'))
        elif name in {'REDUCE','NEWOBJ','NEWOBJ_EX'}:
            count=2 if name in {'REDUCE','NEWOBJ'} else 3
            pieces=[pop_token() for _ in range(count)]
            stack.append(new(min(t.origin for t in pieces),'opaque'))
        elif name=='BUILD':
            pop_token()
            if not stack or not isinstance(stack[-1],_Token):
                raise RuntimeError('Malformed pickle BUILD')
        elif name=='BINPERSID':
            item=pop_token()
            stack.append(new(item.origin,'opaque'))
        elif name=='SETITEM':
            value=pop_token()
            key=pop_token()
            if not stack or not isinstance(stack[-1],_Token):
                raise RuntimeError('Malformed pickle SETITEM')
            if is_root(stack[-1]):
                record_pairs([key,value],pos)
        elif name=='SETITEMS':
            i=last_marker()
            if i<1 or not isinstance(stack[i-1],_Token):
                raise RuntimeError('Malformed pickle SETITEMS')
            target=stack[i-1]
            contents=stack[i+1:]
            if is_root(target):
                record_pairs(contents,pos)
            del stack[i:]
        elif name=='APPEND':
            pop_token()
        elif name in {'APPENDS','ADDITEMS'}:
            i=last_marker()
            if i<1 or not isinstance(stack[i-1],_Token):
                raise RuntimeError('Malformed pickle append')
            del stack[i:]
        elif name=='POP':
            stack.pop()
        elif name=='POP_MARK':
            i=last_marker()
            del stack[i:]
        elif name=='DUP':
            top=stack[-1]
            stack.append(new(pos,top.kind,top.literal,top.identity))
        elif name=='STOP':
            if not stack or not is_root(stack[-1]):
                raise RuntimeError('Expected outer checkpoint dictionary at STOP')
        else:
            raise RuntimeError(f'Unsupported pickle opcode {name} during metadata pruning')

    if not required_keys.issubset(found):
        raise RuntimeError(f'Missing required checkpoint fields: {sorted(required_keys-found)}')
    output=bytearray()
    cursor=0
    for start,end,payload in sorted(patches):
        if start<cursor:
            raise RuntimeError('Overlapping checkpoint metadata rewrites')
        output.extend(data[cursor:start])
        output.extend(payload)
        cursor=end
    output.extend(data[cursor:])
    list(pickletools.genops(bytes(output)))
    return bytes(output)


CHECKPOINT_IDENTITIES = {
    "DroneRGBT": {
        "shared_network": {
            "filename": "best.pt", "size_bytes": 157_856_906,
            "sha256": "e810e9e71e1049624c9755bd7fa85625dba97424bbc5f02b0e85f182db5baf30",
            "state_dict_key": "model",
        },
        "depth_controller": {
            "filename": "depth_controller.pt", "size_bytes": 205_680,
            "sha256": "1d68f21d210ff8add9cae3188c76a1604309fd94d83e7eb12509a28d0b74cdf5",
            "state_dict_key": "router",
        },
    },
    "RGBT-CC": {
        "shared_network": {
            "filename": "best.pt", "size_bytes": 157_782_166,
            "sha256": "a6be4941f24c1f682988ff3d4abf357ea3c21e09bf8116091bda4d1d73f12f29",
            "state_dict_key": "model",
        },
        "depth_controller": {
            "filename": "depth_controller.pt", "size_bytes": 206_186,
            "sha256": "ad55765902a4f57a996bbd89ba24ee3e9603da924b0ea371b6782e732e1e963c",
            "state_dict_key": "router_state",
        },
    },
}
RGBTCC_TEACHER_SHA256 = "12b3b0a7f3ebfe91d426c1c127fde50dd49f36e89c3441d1d6baf917305d12b1"
EXPECTED_DEPLOYED_PARAMETERS = 13_122_151
EXPECTED_CONTROLLER_PARAMETERS = 50_564
MANIFEST_FORMAT = "tce-dual-checkpoint-shards-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe checkpoint path in manifest: {name!r}")
    root = root.resolve(strict=True)
    candidate = root / relative
    if candidate.is_symlink():
        raise RuntimeError(f"Missing or unsafe checkpoint file: {name}")
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Checkpoint file is missing: {name}") from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError(f"Missing or unsafe checkpoint file: {name}")
    return path


def _checkpoint_manifest(root: Path, dataset: str) -> tuple[Path, dict]:
    candidates = (root / "checkpoints" / dataset, root / dataset, root)
    directory = next((p for p in candidates if (p / "manifest.json").is_file()), None)
    if directory is None:
        raise FileNotFoundError(f"{dataset} manifest.json not found under {root}")
    directory = directory.resolve(strict=True)
    manifest_path = _safe_file(directory, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") == "tce-inference-only-CANDIDATE-v1":
        # Derived from the verified frozen checkpoint, with tensor values unchanged.
        pinned = CHECKPOINT_IDENTITIES[dataset]
        shared = manifest.get("shared_network")
        controller = manifest.get("depth_controller")
        if (manifest.get("dataset") != dataset
                or manifest.get("original_shared_sha256") != pinned["shared_network"]["sha256"]
                or not isinstance(shared, dict) or not isinstance(controller, dict)):
            raise RuntimeError(f"{dataset} minimal checkpoint provenance mismatch")
        for key, value in pinned["depth_controller"].items():
            if controller.get(key) != value:
                raise RuntimeError(f"{dataset} minimal checkpoint controller identity mismatch: {key}")
        if (shared.get("filename") != "best.pt" or shared.get("state_dict_key") != "model"
                or not isinstance(shared.get("parts"), list)
                or len(shared["parts"]) != 3):
            raise RuntimeError(f"{dataset} minimal checkpoint shard format mismatch")
        valid = manifest.get("validation") or {}
        if (valid.get("all_source_parameters_and_buffers_equal") is not True
                or valid.get("exported_state_dict_exact_reload") is not True
                or valid.get("original_controller_sha256_unchanged") is not True):
            raise RuntimeError(f"{dataset} minimal export verification missing")
        if not isinstance(manifest.get("source_manifest_sha256"), str):
            raise RuntimeError(f"{dataset} original manifest provenance missing")
        return directory, manifest
    if (manifest.get("format") != MANIFEST_FORMAT or manifest.get("version") != 1
            or manifest.get("dataset") != dataset):
        raise RuntimeError(f"{dataset} checkpoint manifest identity mismatch")
    expected = CHECKPOINT_IDENTITIES[dataset]
    for role in ("shared_network", "depth_controller"):
        spec = manifest.get(role)
        if not isinstance(spec, dict):
            raise RuntimeError(f"{dataset} manifest lacks {role}")
        for key, value in expected[role].items():
            if spec.get(key) != value:
                raise RuntimeError(f"{dataset} {role} {key} mismatch")
    if not isinstance(manifest["shared_network"].get("parts"), list):
        raise RuntimeError(f"{dataset} shared-network part inventory is missing")
    if manifest["depth_controller"].get("parts"):
        raise RuntimeError(f"{dataset} controller must be the verified unsplit checkpoint")
    return directory, manifest


@contextlib.contextmanager
def _verified_checkpoint_path(directory: Path, spec: dict, label: str):
    size = spec.get("size_bytes")
    digest = spec.get("sha256")
    if not isinstance(size, int) or size <= 0 or not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"Invalid {label} identity in manifest")
    parts = spec.get("parts")
    if not parts:
        path = _safe_file(directory, spec.get("filename", ""))
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError(f"{label} size/SHA256 mismatch")
        yield path
        return
    if not isinstance(parts, list):
        raise RuntimeError(f"Invalid {label} part inventory")
    expected_name = spec.get("filename")
    total = 0
    verified = []
    for index, part in enumerate(parts, 1):
        if not isinstance(part, dict):
            raise RuntimeError(f"Invalid {label} part record")
        name = part.get("filename", "")
        if name != f"shared_network/{expected_name}.part{index:03d}":
            raise RuntimeError(f"{label} shard sequence mismatch at part {index}")
        part_size, part_sha = part.get("size_bytes"), part.get("sha256")
        if (not isinstance(part_size, int) or not 0 < part_size < 25_000_000
                or not isinstance(part_sha, str) or len(part_sha) != 64):
            raise RuntimeError(f"Invalid {label} part identity: {name}")
        path = _safe_file(directory, name)
        if path.stat().st_size != part_size or sha256_file(path) != part_sha:
            raise RuntimeError(f"{label} part size/SHA256 mismatch: {name}")
        verified.append(path)
        total += part_size
    if total != size:
        raise RuntimeError(f"{label} reconstructed size mismatch in manifest")
    with tempfile.TemporaryDirectory(prefix="tce-checkpoint-") as temporary:
        restored = Path(temporary) / expected_name
        full_digest = hashlib.sha256()
        written = 0
        with restored.open("xb") as output:
            for part in verified:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        full_digest.update(block)
                        written += len(block)
        if written != size or full_digest.hexdigest() != digest:
            raise RuntimeError(f"{label} reconstructed checkpoint SHA256 mismatch")
        yield restored


def _strict_load(module, state: dict, label: str) -> None:
    if not isinstance(state, dict):
        raise RuntimeError(f"{label} state_dict is missing or invalid")
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{label} strict load reported parameter discrepancies")


def _safe_weights_only_load(torch, path: Path):
    allowed = [pathlib.PosixPath]
    try:
        from torch.torch_version import TorchVersion
    except ImportError:
        TorchVersion = None
    if TorchVersion is not None:
        allowed.append(TorchVersion)
    manager = (torch.serialization.safe_globals(allowed)
               if hasattr(torch.serialization, "safe_globals")
               else contextlib.nullcontext())
    with manager:
        return torch.load(path, map_location="cpu", weights_only=True)


def _build_model():
    from models.backbone import FusionPro
    from models.model import MultiDepthFusionNetwork

    base = FusionPro(pretrained_backbone=False, fusion_mode="fuse", luyou_en=False,
                     res18backbone=True, MSSM_en=False, MSFM_en=True)
    model = MultiDepthFusionNetwork(base)
    if sum(p.numel() for p in model.parameters()) != EXPECTED_DEPLOYED_PARAMETERS:
        raise RuntimeError("Unified deployment parameter count changed")
    controller_parameters = sum(p.numel() for p in model.router.parameters())
    if controller_parameters != EXPECTED_CONTROLLER_PARAMETERS:
        raise RuntimeError("Fusion Depth Controller parameter count changed")
    model.assert_deployment_parameter_boundary()
    return model


def _inference_only_stream(path: Path, retained_keys: frozenset[str]):
    """Prune only unneeded root metadata without evaluating the pickle."""
    if not zipfile.is_zipfile(path):
        raise RuntimeError("Frozen checkpoint must be a torch.save ZIP archive")
    with zipfile.ZipFile(path) as source:
        candidates = [name for name in source.namelist() if name.endswith("/data.pkl")]
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one PyTorch data.pkl member")
        pickle_name = candidates[0]
        cleaned = _prune_unneeded_root_values(source.read(pickle_name), set(retained_keys))
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w", allowZip64=True) as destination:
            for info in source.infolist():
                if info.filename == pickle_name:
                    destination.writestr(info, cleaned)
                else:
                    with (source.open(info) as src,
                          destination.open(info, "w", force_zip64=True) as dst):
                        shutil.copyfileobj(src, dst, 1024 * 1024)
        result.seek(0)
        return result


def _verified_drone_object(path: Path, spec: dict, retained_keys: frozenset[str]):
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Frozen checkpoint missing: {path}")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise RuntimeError(f"Checkpoint byte size mismatch: {path.name}")
    if sha256_file(path).lower() != str(spec["sha256"]).lower():
        raise RuntimeError(f"Frozen checkpoint SHA256 mismatch: {path.name}")
    policies = (frozenset({"model", "epoch"}),
                frozenset({"router", "mode", "cost_weight"}))
    if retained_keys not in policies:
        raise ValueError("Only predefined DroneRGBT inference field policies are accepted")
    try:
        value = torch.load(_inference_only_stream(path, retained_keys), map_location="cpu",
                           weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Restricted safe loading failed for {path.name}: {exc}") from exc
    if not isinstance(value, dict) or not retained_keys.issubset(value):
        raise RuntimeError(f"Checkpoint is missing required inference fields: {path.name}")
    return value


def _load_minimal_from_manifest(checkpoint_dir: Path, manifest: dict):
    """Safely load inference-only FP32 states and original frozen controller."""
    import torch

    dataset = manifest["dataset"]
    shared_spec = manifest["shared_network"]
    controller_spec = manifest["depth_controller"]
    with _verified_checkpoint_path(
        checkpoint_dir, shared_spec, f"{dataset} inference-only shared network"
    ) as shared_path:
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
    if (not isinstance(shared, dict) or set(shared.keys()) != {"model"}
            or not isinstance(shared["model"], dict)):
        raise RuntimeError(f"{dataset} invalid inference-only checkpoint payload")
    if dataset == "DroneRGBT":
        with _verified_checkpoint_path(
            checkpoint_dir, controller_spec, "DroneRGBT Fusion Depth Controller"
        ) as controller_path:
            controller = _verified_drone_object(
                controller_path, controller_spec,
                frozenset({"router", "mode", "cost_weight"}),
            )
        if (controller.get("mode") != "full"
                or float(controller.get("cost_weight", -999)) != 0.0):
            raise RuntimeError("DroneRGBT controller frozen metadata mismatch")
        router_state = controller.get("router")
    elif dataset == "RGBT-CC":
        with _verified_checkpoint_path(
            checkpoint_dir, controller_spec, "RGBT-CC Fusion Depth Controller"
        ) as controller_path:
            controller = _safe_weights_only_load(torch, controller_path)
        if not isinstance(controller, dict):
            raise RuntimeError("RGBT-CC frozen controller metadata missing")
        provenance = controller.get("protocol") or {}
        if (provenance.get("teacher_sha256") != RGBTCC_TEACHER_SHA256
                or provenance.get("student_sha256") != manifest["original_shared_sha256"]):
            raise RuntimeError("RGBT-CC controller original provenance mismatch")
        router_state = controller.get("router_state")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model = _build_model()
    _strict_load(model, shared["model"], f"{dataset} minimal shared network")
    _strict_load(model.router, router_state, f"{dataset} frozen Fusion Depth Controller")
    return model, {
        "protocol": f"{dataset} inference-only FP32 checkpoint, source-verified",
        "checkpoint_sha256": {
            "shared": shared_spec["sha256"], "controller": controller_spec["sha256"]
        },
        "original_shared_sha256": manifest["original_shared_sha256"],
    }


def _load_drone(root: Path):
    checkpoint_dir, manifest = _checkpoint_manifest(root, "DroneRGBT")
    if manifest["format"] == "tce-inference-only-CANDIDATE-v1":
        return _load_minimal_from_manifest(checkpoint_dir, manifest)
    shared_spec, controller_spec = manifest["shared_network"], manifest["depth_controller"]
    with _verified_checkpoint_path(
        checkpoint_dir, shared_spec, "DroneRGBT shared network"
    ) as shared_path:
        shared = _verified_drone_object(
            shared_path, shared_spec, frozenset({"model", "epoch"})
        )
    with _verified_checkpoint_path(
        checkpoint_dir, controller_spec, "DroneRGBT Fusion Depth Controller"
    ) as controller_path:
        controller = _verified_drone_object(
            controller_path, controller_spec,
            frozenset({"router", "mode", "cost_weight"}),
        )
    if int(shared.get("epoch", -1)) != 26:
        raise RuntimeError("DroneRGBT shared-network frozen epoch mismatch")
    if controller.get("mode") != "full" or float(controller.get("cost_weight", -999)) != 0.0:
        raise RuntimeError("DroneRGBT controller metadata differs from the verified original")
    model = _build_model()
    _strict_load(model, shared.get(shared_spec["state_dict_key"]), "DroneRGBT shared network")
    _strict_load(model.router, controller.get(controller_spec["state_dict_key"]),
                 "DroneRGBT Fusion Depth Controller")
    return model, {
        "protocol": "DroneRGBT Reviewer v2.5 restricted metadata pruning with lossless shards",
        "checkpoint_sha256": {
            "shared": shared_spec["sha256"], "controller": controller_spec["sha256"]
        },
    }


def _load_rgbtcc(root: Path):
    import torch

    checkpoint_dir, manifest = _checkpoint_manifest(root, "RGBT-CC")
    if manifest["format"] == "tce-inference-only-CANDIDATE-v1":
        return _load_minimal_from_manifest(checkpoint_dir, manifest)
    shared_spec, controller_spec = manifest["shared_network"], manifest["depth_controller"]
    with _verified_checkpoint_path(
        checkpoint_dir, shared_spec, "RGBT-CC shared network"
    ) as shared_path:
        shared = _safe_weights_only_load(torch, shared_path)
    with _verified_checkpoint_path(
        checkpoint_dir, controller_spec, "RGBT-CC Fusion Depth Controller"
    ) as controller_path:
        controller = _safe_weights_only_load(torch, controller_path)
    if not isinstance(shared, dict) or not isinstance(shared.get("model"), dict):
        raise RuntimeError("Invalid RGBT-CC shared-network checkpoint")
    if not isinstance(controller, dict) or not isinstance(controller.get("router_state"), dict):
        raise RuntimeError("Invalid RGBT-CC Fusion Depth Controller checkpoint")
    student_protocol = shared.get("protocol") or {}
    controller_protocol = controller.get("protocol") or {}
    if student_protocol.get("teacher_sha256") != RGBTCC_TEACHER_SHA256:
        raise RuntimeError("RGBT-CC shared-network initialization identity mismatch")
    if controller_protocol.get("teacher_sha256") != RGBTCC_TEACHER_SHA256:
        raise RuntimeError("RGBT-CC controller initialization identity mismatch")
    if controller_protocol.get("student_sha256") != shared_spec["sha256"]:
        raise RuntimeError("RGBT-CC controller/shared-network identity mismatch")
    if student_protocol.get("kd_weight") != 2.0:
        raise RuntimeError("RGBT-CC frozen self-guidance weight mismatch")
    model = _build_model()
    _strict_load(model, shared["model"], "RGBT-CC shared network")
    _strict_load(model.router, controller["router_state"], "RGBT-CC Fusion Depth Controller")
    return model, {
        "protocol": "RGBT-CC Reviewer Standalone v2 lossless dual-checkpoint shards",
        "checkpoint_sha256": {
            "shared": shared_spec["sha256"], "controller": controller_spec["sha256"]
        },
        "teacher_provenance_sha256": RGBTCC_TEACHER_SHA256,
    }


def load_model(dataset: str, checkpoints_root: str | Path, device: str = "cpu"):
    """Construct the frozen Stage 1 model and strictly load dataset weights."""
    import torch

    root = Path(checkpoints_root).expanduser().resolve(strict=True)
    if dataset == "DroneRGBT":
        model, metadata = _load_drone(root)
    elif dataset == "RGBT-CC":
        model, metadata = _load_rgbtcc(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    model.requires_grad_(False).eval().to(target)
    model.assert_deployment_parameter_boundary()
    return model, metadata
