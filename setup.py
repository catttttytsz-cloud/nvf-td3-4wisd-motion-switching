from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'neural_vector_field'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    scripts=glob('scripts/*.sh'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Shuzhong Tan',
    maintainer_email='your.email@example.com',
    description='Neural vector field TD3 controller for 4WISD mode-switch command generation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_vector_field_policy = neural_vector_field.train_vector_field_policy:main',
            'test_vector_field_policy = neural_vector_field.test_vector_field_policy:main',
        ],
    },
)
