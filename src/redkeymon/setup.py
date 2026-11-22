from setuptools import setup, find_packages
setup(name="hw5pkg", version="0.1", packages=find_packages(),
      install_requires=["torch", "numpy", "scikit-learn", "matplotlib", "onnx", "onnxruntime"])