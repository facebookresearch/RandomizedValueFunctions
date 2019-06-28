# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import setup, find_packages
setup(name='qlearn',
      packages=[package for package in find_packages()
           if package.startswith('qlearn')],
      version='0.1')
