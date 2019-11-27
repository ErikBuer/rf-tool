from setuptools import setup

setup(name='rf-tool',
      version='0.0.1',
      description='RF electronics calculator',
      url='https://github.com/ErikBuer/rf-tool',
      author='Erik Buer',
      author_email='',
      license='GPL',
      packages=['rf-tool'],
      zip_safe=False,
      install_requires     = [ 'numpy','scipy'])