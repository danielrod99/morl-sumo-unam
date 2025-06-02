from setuptools import setup, find_packages

setup(
    name='unam-traffic-rl',
    version='0.1.0',
    author='Daniel Rodríguez',
    description='Multi-objective reinforcement learning for traffic light control near UNAM using SUMO and Pareto Q-Learning.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'sumo-rl',
        'gymnasium',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
