from setuptools import setup, find_packages

setup(
    name="mace-model-deviation",
    version="0.1.0",
    description="Standalone MACE model ensemble uncertainty calculation",
    author="AI2-Kit Contributors",
    author_email="your-email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.11.0",
        "ase>=3.22.0", 
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "mace-model-devi=mace_model_deviation.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)