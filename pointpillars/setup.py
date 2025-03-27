from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch

ext_modules = []

# Check if CUDA is available
if torch.cuda.is_available():
    from torch.utils.cpp_extension import CUDAExtension
    ext_modules = [
        CUDAExtension(
            name='pointpillars.ops.voxel_op',
            sources=[
                'pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars/ops/voxelization/voxelization_cpu.cpp',
                'pointpillars/ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='pointpillars.ops.iou3d_op',
            sources=[
                'pointpillars/ops/iou3d/iou3d.cpp',
                'pointpillars/ops/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ]
else:
    print("⚠️ CUDA not available, building CPU version only.")
    ext_modules = [
        CppExtension(
            name='pointpillars.ops.voxel_op',
            sources=[
                'pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars/ops/voxelization/voxelization_cpu.cpp',
            ],
        ),
        CppExtension(
            name='pointpillars.ops.iou3d_op',
            sources=[
                'pointpillars/ops/iou3d/iou3d.cpp',
            ],
        )
    ]

setup(
    name='pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)
#use the below code
#python setup.py install
#python -c "import pointpillars; print('✅ PointPillars CPU version installed successfully!')"

