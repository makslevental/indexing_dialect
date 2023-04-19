import contextlib
from typing import Optional

from mlir_indexing.ir import Context, Module, InsertionPoint, Location


@contextlib.contextmanager
def mlir_mod_ctx(src: Optional[str] = None,
                 context: Context = None,
                 location: Location = None):
    if context is None:
        context = Context()
    if location is None:
        location = Location.unknown()
    with context, location:
        if src is not None:
            module = Module.parse(src)
        else:
            module = Module.create()
        with InsertionPoint(module.body):
            yield module
