# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
from collections import OrderedDict

import torch


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def resume(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            raise IOError(f"No Checkpoint file found on {f}")
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        # return any further checkpoint data
        return checkpoint

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, except_keys=None):
        load_state_dict(self.model, checkpoint.pop("model"), except_keys)


def check_key(key, except_keys):
    if except_keys is None:
        return False
    else:
        for except_key in except_keys:
            if except_key in key:
                return True
        return False


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger("PersonSearch.checkpoint")
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if check_key(key, except_keys):
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, except_keys=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys)

    # use strict loading
    model.load_state_dict(model_state_dict)


def _unwrap_checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("HAM checkpoint must be a dict or contain a state_dict-like dict.")

    for key in ("state_dict", "model", "module"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    tensor_items = OrderedDict((k, v) for k, v in checkpoint.items() if torch.is_tensor(v))
    if tensor_items:
        return tensor_items

    raise KeyError("Could not find tensor weights in HAM checkpoint.")


def _normalize_ham_key(key):
    prefixes = (
        "module.",
        "model.",
        "state_dict.",
        "base.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _ham_key_candidates(key):
    key = _normalize_ham_key(key)
    candidates = [key]

    if key.startswith("base_model."):
        candidates.append(key[len("base_model."):])
    else:
        candidates.append("base_model." + key)

    if key.startswith("clip_model."):
        candidates.append("base_model." + key[len("clip_model."):])
    if key.startswith("clip."):
        candidates.append("base_model." + key[len("clip."):])

    # Keep order while dropping duplicates.
    return list(OrderedDict((candidate, None) for candidate in candidates).keys())


def load_ham_pretrained(model, ckpt_path, logger=None):
    if logger is None:
        logger = logging.getLogger("IRRA.checkpoint")
    if not ckpt_path:
        logger.info("No HAM pretrained checkpoint provided.")
        return {"loaded": 0, "skipped": 0, "missing": 0, "unexpected": 0}
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("HAM pretrained checkpoint not found: {}".format(ckpt_path))

    logger.info("Loading HAM pretrained checkpoint from {}".format(ckpt_path))
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    loaded_state_dict = _unwrap_checkpoint_state_dict(checkpoint)

    model_state_dict = model.state_dict()
    matched_state_dict = OrderedDict()
    loaded_pairs = []
    skipped = []

    for old_key, value in loaded_state_dict.items():
        if not torch.is_tensor(value):
            skipped.append((old_key, "not a tensor"))
            continue

        match_key = None
        for candidate in _ham_key_candidates(old_key):
            if candidate in model_state_dict and model_state_dict[candidate].shape == value.shape:
                match_key = candidate
                break

        if match_key is None:
            skipped.append((old_key, tuple(value.shape)))
            continue

        matched_state_dict[match_key] = value
        loaded_pairs.append((match_key, old_key, tuple(value.shape)))

    current_state_dict = model_state_dict.copy()
    current_state_dict.update(matched_state_dict)
    model.load_state_dict(current_state_dict, strict=True)

    missing = [key for key in model_state_dict.keys() if key not in matched_state_dict]
    unexpected = [
        key for key in loaded_state_dict.keys()
        if not any(candidate in matched_state_dict for candidate in _ham_key_candidates(key))
    ]

    logger.info(
        "HAM pretrained summary: loaded={}, skipped={}, missing={}, unexpected={}".format(
            len(loaded_pairs), len(skipped), len(missing), len(unexpected)
        )
    )
    for key, old_key, shape in loaded_pairs[:20]:
        logger.info("HAM loaded: {} <= {} shape={}".format(key, old_key, shape))
    if len(loaded_pairs) > 20:
        logger.info("HAM loaded: ... {} more tensors".format(len(loaded_pairs) - 20))
    for key, reason in skipped[:20]:
        logger.info("HAM skipped: {} reason={}".format(key, reason))
    if len(skipped) > 20:
        logger.info("HAM skipped: ... {} more tensors".format(len(skipped) - 20))

    if len(loaded_pairs) == 0:
        logger.warning("No tensors were loaded from HAM checkpoint. Check checkpoint keys and model architecture.")

    return {
        "loaded": len(loaded_pairs),
        "skipped": len(skipped),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
