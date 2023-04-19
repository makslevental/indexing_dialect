//===- IteratorsExtension.cpp - Extension module --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include <string>

#include "indexing-c/Dialects.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <mlir-c/BuiltinTypes.h>
#include <vector>

namespace py = pybind11;
using namespace mlir::python::adaptors;

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

PYBIND11_MODULE(_indexingDialects, mainModule) {
  //===--------------------------------------------------------------------===//
  // Indexing dialect.
  //===--------------------------------------------------------------------===//
  auto indexingModule = mainModule.def_submodule("indexing");

  //
  // Dialect
  //

  indexingModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__indexing__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //
  // Types
  //

  (void)mlir_value_subclass(indexingModule, "TensorValue", [](MlirValue value) {
    return mlirIsATensorValue(value);
  });

  mlir_type_subclass(indexingModule, "IndexTensorType",
                     [](MlirType type) {
                       return mlirTypeIsATensor(type) &&
                              mlirTypeIsAIndex(
                                  mlirShapedTypeGetElementType(type));
                     })
      .def_classmethod(
          "get",
          [](const py::object &cls, const std::vector<int64_t> &shape,
             MlirContext context) {
            return cls(mlirRankedTensorTypeGet(shape.size(), shape.data(),
                                               mlirIndexTypeGet(context),
                                               mlirAttributeGetNull()));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

  (void)mlir_value_subclass(indexingModule, "ArithValue", [](MlirValue value) {
    return mlirIsAnArithValue(value);
  });
}
