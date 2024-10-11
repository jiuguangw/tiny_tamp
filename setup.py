"""Setup script."""

from setuptools import setup

setup(
    name="tiny_tamp",
    version="0.1.0",
    packages=["tiny_tamp"],
    include_package_data=True,
    install_requires=[
        "pybullet==3.2.6",
        "scipy",
        "zmq",
        "numpy==1.26.4",
        "trimesh",
        "black==23.9.1",
        "docformatter==1.7.5",
        "isort==5.12.0",
    ],
)
