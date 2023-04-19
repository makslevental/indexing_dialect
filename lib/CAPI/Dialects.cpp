//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "indexing-c/Dialects.h"

#include "indexing/Dialect/Indexing/IR/Indexing.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include <mlir-c/BuiltinTypes.h>
#include <mlir/CAPI/Support.h>

using namespace mlir;
using namespace mlir::indexing;

//===----------------------------------------------------------------------===//
// Indexing dialect and attributes
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Indexing, indexing, IndexingDialect)

bool mlirTypeIsAIndexingCustom(MlirType type) {
  return unwrap(type).isa<CustomType>();
}

MlirType mlirIndexingCustomTypeGet(MlirContext ctx, MlirStringRef str) {
  return wrap(CustomType::get(unwrap(ctx), unwrap(str)));
}

bool mlirIsATensorValue(MlirValue value) {
  return mlirTypeIsATensor(mlirValueGetType(value));
}

bool mlirIsAnArithValue(MlirValue value) {
  MlirType type = mlirValueGetType(value);
  return mlirTypeIsABF16(type) || mlirTypeIsAComplex(type) ||
         mlirTypeIsAF16(type) || mlirTypeIsAF32(type) || mlirTypeIsAF64(type) ||
         mlirTypeIsAInteger(type) || mlirTypeIsAIndex(type);
}