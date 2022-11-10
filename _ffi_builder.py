
from pathlib import Path

from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef(
    open(Path(__file__).absolute().parent / "include" / "nn.h").read()
)

ffibuilder.set_source("_nn_cffi",
    "#include <nn.h>",
    include_dirs=["include"],
    sources=["src/nn.c"],
    )

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
