add_llvm_library(IndexingPlugin
  # BUILDTREE_ONLY is only for testing purposes
  MODULE BUILDTREE_ONLY
  indexing-plugin.cpp
  
  DEPENDS
  MLIRIndexing
  PLUGIN_TOOL
  mlir-opt
  
  LINK_LIBS
  MLIRIndexing
)
target_include_directories(
  IndexingPlugin
  PRIVATE
  "${INDEXING_BINARY_DIR}/include"
)
