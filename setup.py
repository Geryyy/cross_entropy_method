from setuptools import find_packages, setup


setup(
    name="cross-entropy-method",
    version="0.1.0",
    description="Simple cross entropy method optimizer.",
    packages=find_packages(),
    py_modules=["cem"],
    install_requires=["numpy"],
)
