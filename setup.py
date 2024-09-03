"""Setup script."""

from setuptools import setup

setup(
    name="tiny_tamp",
    version="0.1.0",
    packages=["tiny_tamp"],
    include_package_data=True,
    install_requires=["pybullet", "scipy", "zmq", "numpy", "trimesh"],
)
