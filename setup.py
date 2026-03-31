from setuptools import find_packages, setup

package_name = 'adaptive_path_RL'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot_state_publisher.launch.py',
                                                'launch/spawn_turtlebot3.launch.py',
                                                'launch/main.launch.py']),
        ('share/' + package_name + '/worlds', ['worlds/maze.sdf']),
        ('share/' + package_name + '/maps', ['maps/map.pgm', 'maps/map.yaml']),
        ('share/' + package_name + '/rviz', ['rviz/def.rvizs']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sa',
    maintainer_email='satenikak@yandex.ru',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
