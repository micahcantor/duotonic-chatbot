from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['torch', 'pytorch-ignite', 'transformers==2.5.1', 'tensorboardX==1.8', 'tensorflow', 'spacy']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)