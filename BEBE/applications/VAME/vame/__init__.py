#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""
import sys
sys.dont_write_bytecode = True

from BEBE.applications.VAME.vame.initialize_project import init_new_project
from BEBE.applications.VAME.vame.model import create_trainset
from BEBE.applications.VAME.vame.model import train_model
from BEBE.applications.VAME.vame.model import evaluate_model
from BEBE.applications.VAME.vame.analysis import pose_segmentation
from BEBE.applications.VAME.vame.analysis import motif_videos
from BEBE.applications.VAME.vame.analysis import community
from BEBE.applications.VAME.vame.analysis import community_videos
from BEBE.applications.VAME.vame.analysis import visualization
from BEBE.applications.VAME.vame.analysis import generative_model
from BEBE.applications.VAME.vame.analysis import gif
from BEBE.applications.VAME.vame.util.csv_to_npy import csv_to_numpy
from BEBE.applications.VAME.vame.util.align_egocentrical import egocentric_alignment
from BEBE.applications.VAME.vame.util import auxiliary
from BEBE.applications.VAME.vame.util.auxiliary import update_config

