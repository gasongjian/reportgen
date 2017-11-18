# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='reportgen',
    version=0.1.0,
    description=(
        '一个能自动生成PPTX报表的工具'
    ),
    long_description=open('README.rst').read(),
    author='JSong',
    author_email='gasongjian@126.com',
    maintainer='JSong',
    maintainer_email='gasongjian@126.com',
    license='BSD License',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/gasongjian/reportgen',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.x',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
	install_requires=[
        'pandas',
        'numpy',
        'pptx',
		'PIL',
		'io',
		'time'
    ]
)
