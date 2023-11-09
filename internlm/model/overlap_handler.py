#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Any, Union

import torch
from torch import nn

from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.core.scheduler import SchedulerHook
from internlm.model.linear import FSTPLinear
from internlm.model.utils import (
    all_gather_raw_bias_memory_pool,
    all_gather_raw_memory_pool,
    comm_queue,
)
from internlm.utils.common import get_current_device


class FSTPOverlapHandler:
    """
    FSTP overlap handler for managing the all-gather and reduce_scatter overlapping.
    """

    def __init__(
        self,
        model: Union[nn.Module, nn.ModuleList],
        process_group,
        is_hybrid_sp: bool = False,
        reorder_bwd_comm: bool = False,
    ) -> None:
        self.process_group = process_group
        self.is_hybrid_sp = is_hybrid_sp
        self.reorder_bwd_comm = reorder_bwd_comm
        self.fstp_wqkvs = []
        self.fstp_outs = []
        self.fstp_modules = []
        self.module_name = ["Wqkv", "out_proj", "w1", "w2", "w3"]
        self.fstp_global_handle = dict()  # key: fstp module; value: module global all-gather op handle
        self.bias_global_handle = dict()  # key: fstp module; value: module bias global all-gather op handle
        self.module_to_index = dict()  # key: fstp module; value: transformer block index
        self.index_to_fstp_modules = dict()  # key: transformer block index; value: fsdp modules
        self.chunks = []
        self.model_checkpoint = gpc.config.model.checkpoint
        self.is_forward = True

        self.reduce_scatter_handlers = {}
        self.zero_const_pool = {}

        self._comm_stream = torch.cuda.Stream()

        if self.reorder_bwd_comm:
            assert self.is_hybrid_sp, "reorder_bwd_comm only support hybrid_sp!"

        # just want to share same for loop for ModuleList and Module
        if not isinstance(model, nn.ModuleList):
            model = [model]

        def _attach_fstp_linear(name, child, reduce_scatter_name):
            if name == "Wqkv":
                self.fstp_wqkvs.append(child)
                self.module_to_index[child] = intern_idx
            if name == "out_proj":
                self.fstp_outs.append(child)
                self.module_to_index[child] = intern_idx
            if isinstance(child, FSTPLinear):
                self.module_to_index[child] = intern_idx
                self.fstp_modules.append(child)
                self.index_to_fstp_modules[intern_idx].append(child)

                setattr(child, "_fstp_name", name)

                setattr(child.weight, "_fstp_reduce_scatter_str", f"{reduce_scatter_name}.weight")
                if child.bias is not None:
                    setattr(child.bias, "_fstp_reduce_scatter_str", f"{reduce_scatter_name}.bias")

        for _chunk in model:
            if isinstance(_chunk, NaiveAMPModel):
                _chunk = _chunk.model
            self.chunks.append(_chunk)

            intern_idx = 0
            for _chunk_name, children in _chunk.named_children():
                if not isinstance(children, nn.ModuleList):
                    continue

                for idx, block in enumerate(children):
                    # only process block with intern sp mode
                    if idx % 2 == 1:
                        continue

                    self.index_to_fstp_modules[intern_idx] = []
                    for _sub_name, sub in block.named_children():
                        for name, child in sub.named_children():
                            _full_name = f"{_chunk_name}.{intern_idx}.{_sub_name}.{name}"
                            _attach_fstp_linear(name=name, child=child, reduce_scatter_name=_full_name)

                    intern_idx += 1

        self.num_blocks = len(self.index_to_fstp_modules)

        self._initialize_memory_pool()
        self._register_sync_parameters_hook()

    def get_zero_by_shape(self, size: tuple, dtype, device) -> torch.Tensor:
        if size not in self.zero_const_pool:
            self.zero_const_pool[size] = torch.zeros(*size, dtype=dtype, device=device).contiguous()

        return self.zero_const_pool[size]

    def set_forward_mode(self, flag):
        self.is_forward = flag

    def _initialize_module_shape(self):
        hidden_size = gpc.config.HIDDEN_SIZE
        mlp_ratio = gpc.config.MLP_RATIO
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        mlp_hidden_size = 256 * ((mlp_hidden_size + 256 - 1) // 256)

        self.module_shape["Wqkv"] = (3 * hidden_size, hidden_size)
        self.module_shape["out_proj"] = (hidden_size, hidden_size)
        self.module_shape["w1"] = (mlp_hidden_size, hidden_size)
        self.module_shape["w2"] = (mlp_hidden_size, hidden_size)
        self.module_shape["w3"] = (hidden_size, mlp_hidden_size)

    def _initialize_memory_pool(self) -> None:
        # allocate memory pool
        self.all_gather_memory_pool = []
        self.all_gather_bias_memory_pool = []
        self.reduce_scatter_memory_pool = {}
        self.module_shape = {}

        self._initialize_module_shape()
        dtype = gpc.config.model.get("dtype", torch.half)
        device = get_current_device()

        for _ in range(2):
            weight = {}
            for name in self.module_name:
                weight[name] = torch.zeros(self.module_shape[name], dtype=dtype, device=device).contiguous()
            self.all_gather_memory_pool.append(weight)  # containing two groups of block weight

    def clear_memory_pool(self) -> None:
        self.zero_const_pool = {}
        self.reduce_scatter_memory_pool = {}

    def get_all_gather_memory(self, module):
        block_index = self.module_to_index[module]
        return self.all_gather_memory_pool[block_index % 2][module._fstp_name]

    def get_bias_memory(self, module: nn.Module):
        block_index = self.module_to_index[module]
        # if the bias memory pool is empty or module has been not allocated memory
        if len(self.all_gather_bias_memory_pool) == 0:
            for _ in range(2):
                weight = {}
                weight[module._fstp_name] = torch.zeros(
                    self.module_shape[module._fstp_name][0],
                    dtype=gpc.config.model.get("dtype", torch.half),
                    device=get_current_device(),
                ).contiguous()
                self.all_gather_bias_memory_pool.append(weight)
        elif module._fstp_name not in self.all_gather_bias_memory_pool[0]:
            for i in range(2):
                self.all_gather_bias_memory_pool[i][module._fstp_name] = torch.zeros(
                    self.module_shape[module._fstp_name][0],
                    dtype=gpc.config.model.get("dtype", torch.half),
                    device=get_current_device(),
                ).contiguous()

        return self.all_gather_bias_memory_pool[block_index % 2][module._fstp_name]

    def get_reduce_scatter_memory(self, key):
        # if key not in dict
        if key not in self.reduce_scatter_memory_pool:
            self.reduce_scatter_memory_pool[key] = []

        for index, mem_item in enumerate(self.reduce_scatter_memory_pool[key]):
            if mem_item.idle is True:
                self.reduce_scatter_memory_pool[key][index].idle = False
                return self.reduce_scatter_memory_pool[key][index]

        # if the memory pool is all used
        cur_len = len(self.reduce_scatter_memory_pool[key])
        self.reduce_scatter_memory_pool[key].append(
            torch.zeros(key, dtype=gpc.config.model.get("dtype", torch.half), device=get_current_device()).contiguous()
        )
        setattr(self.reduce_scatter_memory_pool[key][cur_len], "idle", False)
        setattr(self.reduce_scatter_memory_pool[key][cur_len], "index", cur_len)
        return self.reduce_scatter_memory_pool[key][cur_len]

    def release_reduce_scatter_memory(self, key, index):
        self.reduce_scatter_memory_pool[key][index].idle = True

    def _all_gather_block_weight_memory_pool(self, block_index: int):
        fstp_modules = self.index_to_fstp_modules[block_index]
        with torch.cuda.stream(self._comm_stream):
            for module in fstp_modules:
                if module.bias is not None:
                    bias_handle = all_gather_raw_bias_memory_pool(
                        module.bias,
                        self.process_group,
                        async_op=True,
                        module=module,
                    )
                    self.bias_global_handle[module] = bias_handle

                weight_handle = all_gather_raw_memory_pool(
                    module.weight,
                    self.process_group,
                    async_op=True,
                    module=module,
                )
                self.fstp_global_handle[module] = weight_handle

    def _register_sync_parameters_hook(self) -> None:
        """
        register forward hooks and backward hooks for fstp modules.
        """

        def _pre_forward_hook_for_first_block(module: nn.Module, inputs: Any):  # pylint: disable=W0613
            self._all_gather_block_weight_memory_pool(0)

        def _pre_backward_hook_for_last_block(module: nn.Module, grad_output):  # pylint: disable=W0613
            if self.is_hybrid_sp or self.model_checkpoint:
                self._all_gather_block_weight_memory_pool(self.num_blocks - 1)
            else:
                first_backward_module = self.fstp_modules[-1]
                with torch.cuda.stream(self._comm_stream):
                    weight_handle = all_gather_raw_memory_pool(
                        first_backward_module.weight,
                        self.process_group,
                        async_op=True,
                        module=first_backward_module,
                    )
                    self.fstp_global_handle[first_backward_module] = weight_handle

        def _prefetch_hook_for_block(module: nn.Module, *args):  # pylint: disable=W0613
            """
            1. 适用于所有情况的forward
            2. 适用于开启了ckpt的backward
            3. 适用于开启sp混合模式的backward
            """
            block_index = self.module_to_index[module]
            if self.is_forward:
                # start the all-gather for next block
                if block_index + 1 < self.num_blocks:
                    self._all_gather_block_weight_memory_pool(block_index + 1)
            else:
                if block_index - 1 >= 0:
                    if self.reorder_bwd_comm:

                        def hack_all_gather():
                            self._all_gather_block_weight_memory_pool(block_index - 1)

                        comm_queue.append(hack_all_gather)  # 交错的sp混合模式可以保证comm_queue中的通信被触发
                    else:
                        self._all_gather_block_weight_memory_pool(block_index - 1)

        def _pre_forward_hook_for_module(module: nn.Module, inputs: Any):  # pylint: disable=W0613
            with torch.cuda.stream(self._comm_stream):
                if module in self.fstp_global_handle:
                    handle = self.fstp_global_handle[module]
                    handle.wait()
                    if module.bias is not None:
                        bias_handle = self.bias_global_handle[module]
                        bias_handle.wait()
                else:
                    weight_handle = all_gather_raw_memory_pool(
                        module.weight,
                        self.process_group,
                        async_op=True,
                        module=module,
                    )
                    self.fstp_global_handle[module] = weight_handle
                    weight_handle.wait()

        def _post_forward_hook_for_module(module: nn.Module, inputs: Any, output: Any):  # pylint: disable=W0613
            if module in self.fstp_global_handle:
                del self.fstp_global_handle[module]

        def _pre_backward_hook_for_module(module: nn.Module, grad_output):  # pylint: disable=W0613
            with torch.cuda.stream(self._comm_stream):
                # wait handle for current module
                if module in self.fstp_global_handle:
                    weight_handle = self.fstp_global_handle[module]
                    weight_handle.wait()
                else:
                    weight_handle = all_gather_raw_memory_pool(
                        module.weight,
                        self.process_group,
                        async_op=True,
                        module=module,
                    )
                    self.fstp_global_handle[module] = weight_handle
                    weight_handle.wait()

                # start the all-gather for next module
                if self.is_hybrid_sp is False and not self.model_checkpoint:
                    module_index = self.fstp_modules.index(module)
                    if module_index - 1 >= 0:
                        next_module = self.fstp_modules[module_index - 1]
                        weight_handle = all_gather_raw_memory_pool(
                            next_module.weight,
                            self.process_group,
                            async_op=True,
                            module=next_module,
                        )
                        self.fstp_global_handle[next_module] = weight_handle

        def _post_backward_hook_for_module(module, grad_input, grad_output):  # pylint: disable=W0613
            if module in self.fstp_global_handle:
                del self.fstp_global_handle[module]

        # 给model注册hook为首尾block预取权重
        # forward
        for _chunk in self.chunks:
            _chunk.register_forward_pre_hook(_pre_forward_hook_for_first_block)
            _chunk.register_full_backward_pre_hook(_pre_backward_hook_for_last_block)

        # 中间block注册hook
        # forward
        for out_proj in self.fstp_outs:
            out_proj.register_forward_pre_hook(_prefetch_hook_for_block)
        for module in self.fstp_modules:
            module.register_forward_pre_hook(_pre_forward_hook_for_module)
            module.register_forward_hook(_post_forward_hook_for_module)
        # backward
        if self.is_hybrid_sp and not self.model_checkpoint:
            for wqkv in self.fstp_wqkvs:
                wqkv.register_full_backward_pre_hook(_prefetch_hook_for_block)
        for module in self.fstp_modules:
            module.register_full_backward_pre_hook(_pre_backward_hook_for_module)
            module.register_full_backward_hook(_post_backward_hook_for_module)

        # forward: block粒度
        # embedding 为第一个 intern block prefetch weight
        # 在当前 intern block(out_proj module）为下一个 intern block prefetch weight
        # 流水线并行时，中间pp stage，第一个intern block手动wait weight
        # 开启activation ckpt时
        # 处理好 block index的步长
        """
        20 / 4 = 5
        0 1 2 3 4
        """

        # backward: block粒度
        # head 为第一个 intern block prefetch weight
        # 在当前 intern block(wqkv module) 为下一个intern block prefetch weight
        # 流水线并行时，中间pp stage，第一个intern block手动wait weight
        # 开启activation ckpt时

        # 以上处理逻辑，目的为了仅针对intern block，简化overlap handler处理逻辑；
        # 但是存在一种情况，开pp时，中间stage，可能需要非 intern block 为 第一个 intern block prefetch weight

        # 考虑到pp和非pp为第一个block prefetch weight的统一性，我们可以以整个model为粒度，
        # 在model forward之前为第一个block prefetch，在model backward之前为第一个block prefetch


class FSTPOverlapSchedulerHook(SchedulerHook):
    """
    SchedulerHook for fstp overlap handler
    """

    def __init__(self, overlap_handler: FSTPOverlapHandler, zero_optim) -> None:
        self._overlap_handler = overlap_handler
        self._zero_optim = zero_optim

    def before_forward(self, scheduler, inputs) -> None:
        self._overlap_handler.set_forward_mode(True)

    def after_forward(self, scheduler, outputs) -> None:
        pass

    def before_criterion(self, scheduler, outputs, label) -> None:
        pass

    def after_criterion(self, scheduler, loss) -> None:
        pass

    def before_backward(self, scheduler, outputs, outputs_grad) -> None:
        self._overlap_handler.set_forward_mode(False)

    def after_backward(self, scheduler, inputs_grad) -> None:
        self._zero_optim.accumulate_left_grads_after_backward()

    def post_helper_func(self, scheduler, outputs, label) -> None:
        pass
