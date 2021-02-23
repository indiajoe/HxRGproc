from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='HxRGproc',
      version='0.3',
      description='Python Tool for Processing HxRG detector data',
      long_description = readme(),
      classifiers=[
          'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      keywords='H2RG HxRG NIR Astronomy Data Reduction',
      url='https://github.com/indiajoe/HxRGproc',
      author='Joe Ninan',
      author_email='indiajoe@gmail.com',
      license='LGPLv3+',
      packages=['HxRGproc','HxRGproc.reduction','HxRGproc.simulator'],
      entry_points = {
          'console_scripts': ['generate_slope_images=HxRGproc.reduction.generate_slope_images:main',
                          'generate_cds_images=HxRGproc.reduction.generate_cds_images:main'],
      },
      install_requires=[
          'numpy',
          'scipy',
          'astropy',
      ],
      include_package_data=True,
      zip_safe=False)
