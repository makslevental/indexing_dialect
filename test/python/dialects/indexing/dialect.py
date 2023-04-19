# RUN: %PYTHON %s | FileCheck %s

from mlir_indexing.dialects import indexing as idx
from mlir_indexing.ir import Context, IntegerType

from mlir_indexing.runtime.util import mlir_mod_ctx


def run(f):
    print("\nTEST:", f.__name__)
    with Context():
        idx.register_dialect()
        f()
    return f


# CHECK-LABEL: TEST: testArithValue
@run
def testArithValue():
    i32 = IntegerType.get_signless(32)
    with mlir_mod_ctx():
        ten = idx.Tensor.empty([10, 10], i32)
        # CHECK: %[[TEN:.*]] = tensor.empty() : tensor<10x10xi32>
        print(ten.owner)
        # CHECK: Tensor(%[[TEN]], tensor<10x10xi32>)
        print(ten)

        v = ten[0, 0]
        # CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[TEN]][%{{.*}}, %{{.*}}] : tensor<10x10xi32>
        print(v.owner)
        # CHECK: Scalar(%[[EXTRACTED]], i32)
        print(v)

        w = v + v
        # CHECK: %[[ADDI:.*]] = arith.addi %[[EXTRACTED]], %[[EXTRACTED]] : i32
        print(w.owner)
        z = w * w
        # CHECK: %[[MULI:.*]] = arith.muli %[[ADDI]], %[[ADDI]] : i32
        print(z.owner)


# CHECK-LABEL: TEST: testTensorValue
@run
def testTensorValue():
    i32 = IntegerType.get_signless(32)
    with mlir_mod_ctx() as module:
        ten = idx.Tensor.empty([10, 10], i32)
        # CHECK: Tensor(%[[TEN]], tensor<10x10xi32>)
        print(ten)

        twenty = ten + ten
        # CHECK: %[[ADD:.*]] = linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<add>} ins(%[[TEN]], %[[TEN]] : tensor<10x10xi32>, tensor<10x10xi32>) outs(%{{.*}} : tensor<10x10xi32>) -> tensor<10x10xi32>
        print(twenty.owner)

        one_hundred = ten * ten
        # CHECK: %[[MUL:.*]] = linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<mul>} ins(%[[TEN]], %[[TEN]] : tensor<10x10xi32>, tensor<10x10xi32>) outs(%{{.*}} : tensor<10x10xi32>) -> tensor<10x10xi32>
        print(one_hundred.owner)

    # CHECK: module {
    # CHECK:   %[[TEN]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:   %[[ONE:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:   %[[ADD]] = linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<add>} ins(%[[TEN]], %[[TEN]] : tensor<10x10xi32>, tensor<10x10xi32>) outs(%[[ONE]] : tensor<10x10xi32>) -> tensor<10x10xi32>
    # CHECK:   %[[THREE:.*]] = tensor.empty() : tensor<10x10xi32>
    # CHECK:   %[[MUL]] = linalg.elemwise_binary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.binary_fn<mul>} ins(%[[TEN]], %[[TEN]] : tensor<10x10xi32>, tensor<10x10xi32>) outs(%[[THREE]] : tensor<10x10xi32>) -> tensor<10x10xi32>
    # CHECK: }
    print(module)


# CHECK-LABEL: TEST: testTensorType
@run
def testTensorType():
    i32 = IntegerType.get_signless(32)
    with mlir_mod_ctx():
        tt = idx.Tensor[(10, 10), i32]
        # CHECK: tensor<10x10xi32>
        print(tt)

        tt = idx.Tensor[(None, None), i32]
        # CHECK: tensor<?x?xi32>
        print(tt)

        tt = idx.IndexTensorType.get([10, 10])
        # CHECK: tensor<10x10xindex>
        print(tt)
