"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

from .latent_factor_models import PureSVD, RSVD, Slim, iALS
from .unpersonalized import Random, MostPop
from .autoencoders import EASER
from .graph_based import RP3beta
from .knn import ItemKNN
from .generic import ProxyRecommender
