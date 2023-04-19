//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_C_DIALECTS_H
#define INDEXING_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Indexing dialect and types
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Indexing, indexing);

MLIR_CAPI_EXPORTED bool mlirTypeIsAIndexingCustom(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirIndexingCustomTypeGet(MlirContext ctx,
                                                      MlirStringRef str);

MLIR_CAPI_EXPORTED bool mlirIsATensorValue(MlirValue value);

MLIR_CAPI_EXPORTED bool mlirIsAnArithValue(MlirValue value);

#ifdef __cplusplus
}
#endif

#endif // INDEXING_C_DIALECTS_H
