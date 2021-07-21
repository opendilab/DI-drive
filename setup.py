from __future__ import absolute_import

from setuptools import setup, find_packages

description = """DI-drive: OpenDILab Deep Learning Autonomous Driving Platform"""

setup(
    name='DI-drive',
    version='0.1',
    description='OpenDILab Deep Learning Autonomous Driving Platform',
    long_description=description,
    author='OpenDILab',
    license='MIT License',
    keywords='DI RL AD Platform',
    packages=[
        *find_packages(
            include=('core', 'core.*')
        ),
    ],
    install_requires=[
        'ephem',
        'h5py',
        'imageio',
        'imgaug',
        'lmdb',
        'loguru==0.3.0',
        'networkx',
        'scipy==1.2.1',
        'pandas',
        'py-trees==0.8.3',
        'pygame==1.9.6',
        'torch>=1.4,<=1.7.1',
        'scikit-image',
        'setuptools==50',
        'shapely',
        'terminaltables',
        'tqdm',
        'xmlschema',
    ]
)
