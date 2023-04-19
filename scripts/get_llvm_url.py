import os
import platform
import sys


def get_llvm_url():
    system = platform.system()
    system_suffix = {
        "Linux": "linux-gnu-ubuntu-22.04",
        "Darwin": "apple-darwin"
    }[system]
    LIB_ARCH = os.environ.get("LIB_ARCH", platform.machine())
    assert LIB_ARCH, "empty LIB_ARCH"
    if LIB_ARCH == "aarch64":
        LIB_ARCH = "arm64"
    # print(f"ARCH {LIB_ARCH}")
    LLVM_RELEASE_VERSION = os.environ.get("LLVM_RELEASE_VERSION", "17.0.0")
    assert LLVM_RELEASE_VERSION, "empty LLVM_RELEASE_VERSION"
    # print(f"ARCH {LIB_ARCH}")
    name = f"llvm+mlir+openmp-{sys.version_info.major}.{sys.version_info.minor}-{LLVM_RELEASE_VERSION}-{LIB_ARCH}-{system_suffix}-release"
    url = f"https://github.com/makslevental/llvm-releases/releases/download/llvm-{LLVM_RELEASE_VERSION}/{name}.tar.xz"
    return url

print(get_llvm_url())