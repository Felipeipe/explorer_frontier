from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'explorer_frontier'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='felipeipe',
    maintainer_email='gonzalezpobletefelipe@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_detection = explorer_frontier.frontier_detection:main',
            'navigator = explorer_frontier.navigator:main'
        ],
    },
)
