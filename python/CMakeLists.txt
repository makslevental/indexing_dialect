include(MLIRDetectPythonEnv)
include(AddMLIRPython)
mlir_configure_python_dev_packages()
mlir_detect_pybind11_install()

# Specifies that all MLIR packages are co-located under the `mlir_indexing`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_indexing.")

# ###############################################################################
# Sources
# ###############################################################################
declare_mlir_python_sources(IndexingPythonSources)
declare_mlir_python_sources(IndexingPythonSources.Dialects
  ADD_TO_PARENT IndexingPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IndexingPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_indexing"
  TD_FILE dialects/IndexingOps.td
  SOURCES
  dialects/indexing.py
  DIALECT_NAME indexing
)

declare_mlir_python_extension(IndexingPythonSources.DialectExtension
  MODULE_NAME _indexingDialects
  ADD_TO_PARENT IndexingPythonSources
  SOURCES
  IndexingDialect.cpp
  EMBED_CAPI_LINK_LIBS
  IndexingCAPI
)

declare_mlir_python_extension(IndexingPythonSources.PassesExtension
  MODULE_NAME _mlirIndexingPasses
  ADD_TO_PARENT IndexingPythonSources
  SOURCES
  IndexingPasses.cpp
  EMBED_CAPI_LINK_LIBS
  IndexingCAPI
)

declare_mlir_python_sources(IndexingPythonSources.ExecutionEngine
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_indexing"
  ADD_TO_PARENT IndexingPythonSources
  SOURCES_GLOB
    runtime/*.py
)

# ###############################################################################
# Common CAPI
# ###############################################################################
add_mlir_python_common_capi_library(IndexingPythonCAPI
  INSTALL_COMPONENT IndexingPythonModules
  INSTALL_DESTINATION python_packages/indexing/mlir_indexing/_mlir_libs
  OUTPUT_DIRECTORY "${INDEXING_BINARY_DIR}/python_packages/mlir_indexing/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
  IndexingPythonSources
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
)

# ###############################################################################
# Instantiation of all Python modules
# ###############################################################################
add_mlir_python_modules(IndexingPythonModules
  ROOT_PREFIX "${INDEXING_BINARY_DIR}/python_packages/mlir_indexing"
  INSTALL_PREFIX "python_packages/mlir_indexing"
  DECLARED_SOURCES
  IndexingPythonSources
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
  IndexingPythonCAPI
)