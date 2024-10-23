from setuptools import setup, find_packages

setup(
    name='aMora',
    version='1.0.0',
    description='Real Estate Pricing Model API',
    author='Júlio',
    packages=find_packages(include=['API', 'API.*']),
    install_requires=[
        'fastapi',
        'httpx',
        'joblib',
        'pandas',
        'pydantic',
        'pytest',
        'python-multipart',
        'scikit-learn'
    ],
    extras_require={
        'dev': [
            'tox',
            'pytest',
            'pytest-asyncio',
            'virtualenv'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
