from setuptools import find_packages, setup


setup(
    name="PyTr2d",
    version="1.0.0",
    url="https://github.com/juglab/PyTr2d.git",
    author="Sheida-rk",
    author_email="95613937+Sheida-RK@users.noreply.github.com",
    description="Python based Tracking in 2D",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "gurobipy",
        "scikit-image",
        "scikit-learn",
        "tifffile",
        "tqdm",
        "py-ctcmetrics",
    ],
)
