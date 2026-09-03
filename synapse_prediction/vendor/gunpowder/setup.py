from setuptools import setup, find_packages

# Vendored fork of gunpowder (1.3.0) used by this pipeline. The key difference
# from upstream is GraphSource(graph_params, graph, graph_spec), which takes
# [db_name, host] and builds the MongoDbGraphProvider internally.
setup(
    name='gunpowder',
    version='1.3.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
