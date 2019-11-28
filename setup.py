from setuptools import setup

setup(name='rf-tool',
      version='0.0.3',
      description='RF electronics calculator',
      url='https://github.com/ErikBuer/rf-tool',
      author='Erik Buer',
      author_email='erik.buer@norskdatateknikk.no',
      license='GPL',
      packages=['rftool'],
      zip_safe=False,
      install_requires     = [ 'numpy','scipy'])