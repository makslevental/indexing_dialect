# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import re
from typing import Union, Tuple

from . import arith as arith_dialect
from . import linalg
from . import tensor as tensor_dialect
# noinspection PyUnresolvedReferences
from ._indexing_ops_gen import *
from .linalg.opdsl.lang.emitter import _BodyBuilder
# noinspection PyUnresolvedReferences
from .._mlir_libs import _mlirIndexingPasses as _cextIndexingPasses
from .._mlir_libs._indexingDialects.indexing import *
from ..ir import Value, Type, RankedTensorType, ShapedType

res_val_reg = re.compile(r"(%\w+) =")

_body_builder = _BodyBuilder({}, {}, {})


class Scalar(ArithValue):

    def __str__(self):
        s = res_val_reg.findall(super().__str__())
        assert len(s) == 1
        return f"Scalar({s[0]}, {self.type})"

    def __add__(self, other) -> "Scalar":
        return Scalar(_body_builder._binary_add(self, other))

    def __sub__(self, other) -> "Scalar":
        return Scalar(_body_builder._binary_sub(self, other))

    def __mul__(self, other) -> "Scalar":
        return Scalar(_body_builder._binary_mul(self, other))


class Tensor(TensorValue):

    def __str__(self):
        s = res_val_reg.findall(super().__str__())
        assert len(s) == 1
        return f"Tensor({s[0]}, {self.type})"

    @property
    def _shaped_type(self) -> ShapedType:
        return ShapedType(self.type)

    @property
    def shape(self) -> list[int]:
        assert self._shaped_type.has_static_shape, "Only static shapes currently supported."
        return self._shaped_type.shape

    @property
    def element_type(self) -> Type:
        return self._shaped_type.element_type

    @classmethod
    def empty(cls, shape: Union[list[int | Value], tuple[int | Value, ...]],
              el_type: Type) -> "Tensor":

        return cls(tensor_dialect.EmptyOp(shape, el_type).result)

    def __class_getitem__(
        cls, dim_sizes_el_type: Tuple[Union[list[int], tuple[int, ...]],
                                      Type]) -> Type:
        assert (len(dim_sizes_el_type) == 2
                ), f"wrong dim_sizes_el_type: {dim_sizes_el_type}"
        dim_sizes, el_type = dim_sizes_el_type
        assert isinstance(el_type,
                          Type), f"wrong type T args for tensor: {el_type}"
        static_sizes = []
        for s in dim_sizes:
            if isinstance(s, int):
                static_sizes.append(s)
            else:
                static_sizes.append(ShapedType.get_dynamic_size())
        return RankedTensorType.get(static_sizes, el_type)

    def __getitem__(self, dims: tuple) -> Scalar:
        dims = list(dims)
        for i, d in enumerate(dims):
            if isinstance(d, int):
                dims[i] = arith_dialect.ConstantOp.create_index(d).result

        return Scalar(tensor_dialect.ExtractOp(self, dims).result)

    def __add__(self, other) -> "Tensor":
        out = Tensor.empty(self.shape, self.element_type)
        return Tensor(
            linalg.elemwise_binary(self,
                                   other,
                                   outs=[out],
                                   fun=linalg.BinaryFn.add))

    def __sub__(self, other) -> "Tensor":
        out = Tensor.empty(self.shape, self.element_type)
        return Tensor(
            linalg.elemwise_binary(self,
                                   other,
                                   outs=[out],
                                   fun=linalg.BinaryFn.sub))

    def __mul__(self, other) -> "Tensor":
        out = Tensor.empty(self.shape, self.element_type)
        return Tensor(
            linalg.elemwise_binary(self,
                                   other,
                                   outs=[out],
                                   fun=linalg.BinaryFn.mul))
