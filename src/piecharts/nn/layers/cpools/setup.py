from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="cpools",
    version="0.1",
    packages=find_packages(),  # Automatically finds Python packages
    ext_modules=[
        CppExtension("cpools.top_pool", ["cpools/top_pool.cpp"]),
        CppExtension("cpools.bottom_pool", ["cpools/bottom_pool.cpp"]),
        CppExtension("cpools.left_pool", ["cpools/left_pool.cpp"]),
        CppExtension("cpools.right_pool", ["cpools/right_pool.cpp"])
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
