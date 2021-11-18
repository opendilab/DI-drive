from __future__ import absolute_import

from setuptools import setup, find_packages

description = """DI-drive: OpenDILab Decision Intelligence Autonomous Driving Platform"""

setup(
    name='DI-drive',
    version='0.2.1',
    description='OpenDILab Decision Intelligence Autonomous Driving Platform',
    long_description=description,
    author='OpenDILab',
    license='MIT License',
    keywords='DL RL AD Platform',
    packages=[
        *find_packages(include=('core', 'core.*')),
    ],
    install_requires=[
        'ephem',
        'h5py',
        'imageio',
        'imgaug',
        'lmdb',
        'loguru==0.3.0',
        'networkx',
        'pandas',
        'py-trees==0.8.3',
        'pygame==1.9.6',
        'torch>=1.4,<=1.8',
        'torchvision',
        'di-engine==0.2',
        'scikit-image',
        'setuptools==50',
        'shapely',
        'terminaltables',
        'tqdm',
        'xmlschema',
    ]
)
