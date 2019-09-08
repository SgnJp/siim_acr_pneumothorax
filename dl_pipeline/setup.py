import setuptools


setuptools.setup(
    name='dl_pipeline',
    version='0.0.1',
    author="Yury Dzerin",
    author_email="yury.dzerin@gmail.com",
    description="A package for training models",
    packages=setuptools.find_packages(),
    install_requires = [
        'torch']
 )
