import torch
import torch.nn as nn
import models.encoders.encoder as enc
import models.aggregators.aggregator as agg
import models.decoders.decoder as dec
from typing import Dict, Union


class PredictionModel(nn.Module):
    """
    Single-agent prediction model
    """
    def __init__(self, encoder: enc.PredictionEncoder,
                 aggregator: agg.PredictionAggregator,
                 decoder: dec.PredictionDecoder):
        """
        Initializes model for single-agent trajectory prediction
        """
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder

    def forward(self, inputs, is_test=False, is_val=False):
        """
        Forward pass for prediction model
        :param inputs: Dictionary with
            'target_agent_representation': target agent history
            'surrounding_agent_representation': surrounding agent history
            'map_representation': HD map representation
        :return outputs: K Predicted trajectories and/or their probabilities
        """
        if is_test:
            encodings, rels = self.encoder(inputs, is_test=is_test)
            agg_encoding = self.aggregator(encodings)
            outputs = self.decoder(agg_encoding)

            return outputs, rels
        
        elif is_val:
            encodings, nei_num, max_dis, mean_dis = self.encoder(inputs, is_val=is_val)
            agg_encoding = self.aggregator(encodings)
            outputs = self.decoder(agg_encoding)

            return outputs, nei_num, max_dis, mean_dis

        else:
            encodings = self.encoder(inputs)
            agg_encoding = self.aggregator(encodings)
            outputs = self.decoder(agg_encoding)

            return outputs
