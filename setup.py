import setuptools
from setuptools import setup
import os 

parent_dir = os.path.dirname(os.path.realpath(__file__))

setup(
    name='GEPrediction-OSRS',
    version='1.0',
    description='GEPrediction ML',
    author='',
    author_email='',
    url='',
    python_requires='>=3.8',
    #package_dir={'GEPrediction-OSRS': parent_dir},
    install_requires=[
        'requests',
        'matplotlib',
        'numpy==1.19.2',
        'sklearn',
        'pandas',
        'flask',
        'imageio',
        'tensorflow'
        ]
    )