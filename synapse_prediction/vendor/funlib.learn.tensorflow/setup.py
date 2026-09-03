from setuptools import setup, find_namespace_packages

# Contains one compiled extension (losses/impl/wrappers*.so) built for
# CPython 3.7 / linux-x86_64. Reuse on a matching platform; otherwise rebuild
# from the upstream funlib.learn.tensorflow source.
setup(
    name='funlib.learn.tensorflow',
    version='0.1',
    packages=find_namespace_packages(include=['funlib.*']),
    package_data={'': ['*.so']},
    include_package_data=True,
    zip_safe=False,
)
