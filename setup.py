from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pyHxRG',
      version='0.3',
      description='Python Tool for Reducing HxRG near-infrared data',
      long_description = readme(),
      classifiers=[
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      keywords='H2RG HxRG NIR Astronomy Data Reduction',
      url='https://github.com/indiajoe/pyHxRG',
      author='Joe Ninan',
      author_email='indiajoe@gmail.com',
      license='LGPLv3+',
      packages=['pyHxRG','pyHxRG.reduction','pyHxRG.simulator'],
      entry_points = {
          'console_scripts': ['generate_slope_imagesT=pyHxRG.reduction.generate_slope_images:main_Teledyne'],
      },
      install_requires=[
          'numpy',
          'astropy',
      ],
      include_package_data=True,
      zip_safe=False)
