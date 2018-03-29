# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='reportgen',
    version='0.1.8',
    description=(
        'reportgen is a Python library for creating and updating analysis report.'
    ),
    long_description=open('README.rst').read(),
    author='JSong',
    author_email='gasongjian@126.com',
    maintainer='JSong',
    maintainer_email='gasongjian@126.com',
    license='BSD License',
    packages=find_packages(),
    include_package_data=True,
        # relative to the vfclust directory
        package_data={
            'images':[
                 'logo.png'],
            'template':
                 ['template.pptx'],
            'font':['readme.txt','DroidSansFallback.ttf']
        },
    platforms=["all"],
    url='https://github.com/gasongjian/reportgen',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
	install_requires=[
        'pandas',
        'numpy',
        'seaborn',
        'python-pptx',
        'Pillow'
    ]
)
