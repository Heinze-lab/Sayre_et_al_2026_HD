from setuptools import setup, find_namespace_packages

setup(
    name='funlib.persistence',
    version='0.1.0',
    packages=find_namespace_packages(include=['funlib.*']),
    include_package_data=True,
    zip_safe=False,
)
