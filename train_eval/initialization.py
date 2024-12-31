# Import datasets
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from datasets.interface import TrajectoryDataset
from datasets.nuScenes.nuScenes_raster import NuScenesRaster
from datasets.nuScenes.nuScenes_vector import NuScenesVector
from datasets.nuScenes.nuScenes_graphs import NuScenesGraphs

# Import models
from models.model import PredictionModel
from models.encoders.raster_encoder import RasterEncoder
from models.encoders.polyline_subgraph import PolylineSubgraphs
from models.encoders.pgp_encoder import PGPEncoder
from models.aggregators.concat import Concat
from models.aggregators.global_attention import GlobalAttention
from models.aggregators.goal_conditioned import GoalConditioned
from models.aggregators.pgp import PGP
from models.decoders.mtp import MTP
from models.decoders.multipath import Multipath
from models.decoders.covernet import CoverNet
from models.decoders.lvm import LVM

#
# from models.encoders.motif_encoder_v1 import Motif_Encoder_V1
# from models.encoders.motif_encoder_v2 import Motif_Encoder_V2
# from models.encoders.motif_encoder_v3 import Motif_Encoder_V3
# from models.encoders.motif_encoder_v4 import Motif_Encoder_V4
from models.encoders.motif_encoder_v5 import Motif_Encoder_V5
from models.encoders.motif_encoder_c0 import Motif_Encoder_C0
from models.encoders.motif_encoder_c1 import Motif_Encoder_C1
from models.encoders.motif_encoder_c3 import Motif_Encoder_C3

from models.encoders.motif_encoder_v510 import Motif_Encoder_V510
from models.encoders.motif_encoder_v6 import Motif_Encoder_V6


# from models.aggregators.motif_agg_v1 import Motif_Agg_V1
from models.aggregators.motif_agg_v5 import Motif_Agg_V5


# from models.decoders.motif_decoder_v1 import Motif_Decoder_V1
from models.decoders.motif_decoder_v5 import Motif_Decoder_V5


# Import metrics
from metrics.mtp_loss import MTPLoss
from metrics.min_ade import MinADEK
from metrics.min_fde import MinFDEK
from metrics.miss_rate import MissRateK
from metrics.covernet_loss import CoverNetLoss
from metrics.pi_bc import PiBehaviorCloning
from metrics.goal_pred_nll import GoalPredictionNLL

from typing import List, Dict, Union


# Datasets
def initialize_dataset(dataset_type: str, args: List) -> TrajectoryDataset:
    """
    Helper function to initialize appropriate dataset by dataset type string
    """
    # TODO: Add more datasets as implemented
    dataset_classes = {'nuScenes_single_agent_raster': NuScenesRaster,
                       'nuScenes_single_agent_vector': NuScenesVector,
                       'nuScenes_single_agent_graphs': NuScenesGraphs,
                       }
    return dataset_classes[dataset_type](*args)


def get_specific_args(dataset_name: str, data_root: str, version: str = None) -> List:
    """
    Helper function to get dataset specific arguments.
    """
    # TODO: Add more datasets as implemented
    specific_args = []
    if dataset_name == 'nuScenes':
        ns = NuScenes(version, dataroot=data_root)
        pred_helper = PredictHelper(ns)
        specific_args.append(pred_helper)

    return specific_args


# Models
def initialize_prediction_model(encoder_type: str, aggregator_type: str, decoder_type: str,
                                encoder_args: Dict, aggregator_args: Union[Dict, None], decoder_args: Dict):
    """
    Helper function to initialize appropriate encoder, aggegator and decoder models
    """
    encoder = initialize_encoder(encoder_type, encoder_args)
    aggregator = initialize_aggregator(aggregator_type, aggregator_args)
    decoder = initialize_decoder(decoder_type, decoder_args)
    model = PredictionModel(encoder, aggregator, decoder)

    return model


def initialize_encoder(encoder_type: str, encoder_args: Dict):
    """
    Initialize appropriate encoder by type.
    """
    # TODO: Update as we add more encoder types
    encoder_mapping = {
        'raster_encoder': RasterEncoder,
        'polyline_subgraphs': PolylineSubgraphs,
        'pgp_encoder': PGPEncoder,
        #'motif_encoder_v1': Motif_Encoder_V1,
        #'motif_encoder_v2': Motif_Encoder_V2,
        #'motif_encoder_v3': Motif_Encoder_V3,
        #'motif_encoder_v4': Motif_Encoder_V4,
        'motif_encoder_v5': Motif_Encoder_V5,
        'motif_encoder_v510': Motif_Encoder_V510,
        'motif_encoder_v6': Motif_Encoder_V6,

        # ab
        'motif_encoder_c0': Motif_Encoder_C0,
        'motif_encoder_c1': Motif_Encoder_C1,
        'motif_encoder_c3': Motif_Encoder_C3,


    }

    return encoder_mapping[encoder_type](encoder_args)


def initialize_aggregator(aggregator_type: str, aggregator_args: Union[Dict, None]):
    """
    Initialize appropriate aggregator by type.
    """
    # TODO: Update as we add more aggregator types
    aggregator_mapping = {
        'concat': Concat,
        'global_attention': GlobalAttention,
        'gc': GoalConditioned,
        'pgp': PGP,
        # 'motif_agg_v1': Motif_Agg_V1,
        'motif_agg_v5': Motif_Agg_V5,
    }

    if aggregator_args:
        return aggregator_mapping[aggregator_type](aggregator_args)
    else:
        return aggregator_mapping[aggregator_type]()


def initialize_decoder(decoder_type: str, decoder_args: Dict):
    """
    Initialize appropriate decoder by type.
    """
    # TODO: Update as we add more decoder types
    decoder_mapping = {
        'mtp': MTP,
        'multipath': Multipath,
        'covernet': CoverNet,
        'lvm': LVM,
        # 'motif_decoder_v1': Motif_Decoder_V1,
        'motif_decoder_v5': Motif_Decoder_V5,
    }

    return decoder_mapping[decoder_type](decoder_args)


# Metrics
def initialize_metric(metric_type: str, metric_args: Dict = None):
    """
    Initialize appropriate metric by type.
    """
    # TODO: Update as we add more metrics
    metric_mapping = {
        'mtp_loss': MTPLoss,
        'covernet_loss': CoverNetLoss,
        'min_ade_k': MinADEK,
        'min_fde_k': MinFDEK,
        'miss_rate_k': MissRateK,
        'pi_bc': PiBehaviorCloning,
        'goal_pred_nll': GoalPredictionNLL
    }

    if metric_args is not None:
        return metric_mapping[metric_type](metric_args)
    else:
        return metric_mapping[metric_type]()
