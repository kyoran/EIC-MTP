from models.encoders.encoder import PredictionEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from typing import Dict
import math
import numpy as np
from motifx.motifx import MotifX

# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #import pdb; pdb.set_trace()
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.initialize_position_embedding(self.position_embedding.weight)

    def initialize_position_embedding(self, weight):
        nn.init.xavier_uniform_(weight)

    def forward(self, x):
        batch_size, seq_len, D = x.shape

        # 生成并取出对应的位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embedding = self.position_embedding(position_ids)
        return x + position_embedding
        

class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, input_dim, max_num):
        super(LearnablePositionalEncoding2D, self).__init__()
        self.input_dim = input_dim
        self.max_num = max_num
        
        # Define learnable parameters
        self.pos_encoding = nn.Parameter(torch.randn(input_dim, max_num, max_num))
        
    def forward(self, input_tensor):
        # Assuming input_tensor has shape [batch_size, input_dim, height, width]
        #import pdb; pdb.set_trace()
        #print("2input_tensor:", input_tensor.shape)
        batch_size, _, nnn, nnn = input_tensor.size()
        try:
            # Repeat positional encoding across the batch
            pos_encoding = self.pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            
            # Add positional encoding to input_tensor
            output_tensor = input_tensor + pos_encoding[:, :, :nnn, :nnn]
        except:
            import pdb; pdb.set_trace()
            
        return output_tensor


class ScalableLearnablePositionalEncoding2D(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.initialize_position_embedding(self.position_embedding.weight)

        
        
    def initialize_position_embedding(self, weight):
        nn.init.xavier_uniform_(weight)

    def forward(self, x):
        batch_size, seq_len, D = x.shape
        
        
        scale_factor = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0) * self.max_len / seq_len
        
        
        # 生成并取出对应的位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embedding = self.position_embedding(position_ids)
        
        return x + position_embedding


class TransformerLayers(nn.Module):
    def __init__(self, d_model, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layers = TransformerEncoderLayer(d_model, num_heads, d_model*mlp_ratio, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src, mask=None):
        L, B, D = src.shape  # seq_len, batch_size, embedding_dim
        # src = src * math.sqrt(self.d_model)
        src = src.view(L, B, D)
        # src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=mask)
        # import pdb; pdb.set_trace()
        output = output.transpose(0, 1).view(B, L, D)
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 这些是q,k,v需要的线性层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False,)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False,)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False,)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 将输入数据拆分成self.heads个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 乘以查询矩阵并计算得分
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # 使用softmax函数得到注意力概率分数
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class Motif_Encoder_V6(PredictionEncoder):

    def __init__(self, args: Dict):
        """
        GRU based encoder from PGP. Lane node features and agent histories encoded using GRUs.
        Additionally, agent-node attention layers infuse each node encoding with nearby agent context.
        Finally GAT layers aggregate local context at each node.

        args to include:

        target_agent_feat_size: int Size of target agent features
        target_agent_emb_size: int Size of target agent embedding
        taret_agent_enc_size: int Size of hidden state of target agent GRU encoder

        node_feat_size: int Size of lane node features
        node_emb_size: int Size of lane node embedding
        node_enc_size: int Size of hidden state of lane node GRU encoder

        nbr_feat_size: int Size of neighboring agent features
        nbr_enb_size: int Size of neighboring agent embeddings
        nbr_enc_size: int Size of hidden state of neighboring agent GRU encoders

        num_gat_layers: int Number of GAT layers to use.
        """

        super().__init__()
        self.args = args

        # Target agent encoder
        self.target_agent_emb = nn.Linear(args['target_agent_feat_size'], args['target_agent_emb_size'])    # [b=32, t=5, feat=5]
        self.target_agent_pos_enc = FixedPositionalEncoding(d_model=args['target_agent_emb_size'], dropout=0.1, max_len=5)
        self.target_agent_tf_enc = TransformerLayers(d_model=args['target_agent_emb_size'], nlayers=2, mlp_ratio=2, num_heads=2, dropout=0.1)
        self.target_agent_tf_norm = nn.LayerNorm(args['target_agent_emb_size'])
        self.target_agent_fc_enc = nn.Linear(args['target_agent_emb_size']*5, args['target_agent_enc_size'])
        # self.target_agent_enc = nn.GRU(args['target_agent_emb_size'], args['target_agent_enc_size'], batch_first=True)

        # Node encoders
        self.node_emb = nn.Linear(args['node_feat_size'], args['node_emb_size'])
        self.node_pos_enc = FixedPositionalEncoding(d_model=args['target_agent_emb_size'], dropout=0.1, max_len=5)
        self.node_encoder = nn.GRU(args['node_emb_size'], args['node_enc_size'], batch_first=True)

        # Surrounding agent encoder
        self.nbr_emb = nn.Linear(args['nbr_feat_size'] + 1, args['nbr_emb_size'])
        self.nbr_pos_enc = FixedPositionalEncoding(d_model=args['target_agent_emb_size'], dropout=0.1, max_len=5)
        self.nbr_tf_enc = TransformerLayers(d_model=args['nbr_emb_size'], nlayers=2, mlp_ratio=2, num_heads=2, dropout=0.1)
        self.nbr_tf_norm = nn.LayerNorm(args['nbr_emb_size'])
        self.nbr_fc_enc = nn.Linear(args['nbr_emb_size'] * 5, args['nbr_enc_size'])
      
      
        latent_pos_max_len = 400
        semantic_pos_max_len = 5000
        # latent relationship
        self.latent_rel_emb = nn.Linear(args['nbr_enc_size'], args['nbr_enc_size'])
        self.latent_rel_pos = LearnablePositionalEncoding(d_model=args['nbr_enc_size'], dropout=0.1, max_len=latent_pos_max_len)
        self.query_latent_rel_emb = nn.Linear(args['target_agent_enc_size'], args['nbr_enc_size'])
        #self.query_latent_rel_pos = LearnablePositionalEncoding(d_model=args['nbr_enc_size'], dropout=0.1, max_len=latent_pos_max_len)
        #self.query_latent_rel_bn = nn.BatchNorm1d(args['nbr_enc_size'])
        self.key_latent_rel_emb = nn.Linear(args['nbr_enc_size'], args['nbr_enc_size'])
        #self.key_latent_rel_pos = LearnablePositionalEncoding(d_model=args['nbr_enc_size'], dropout=0.1, max_len=latent_pos_max_len)
        self.key_latent_rel_bn = nn.BatchNorm1d(args['nbr_enc_size'])

        # latent token
        self.latent_rel_token_emb = nn.Linear(1, args['nbr_enc_size']//2)
        #self.latent_rel_token_bn = nn.BatchNorm1d(args['nbr_enc_size']//2)

        # explicit token
        self.explicit_rel_token_emb = nn.Linear(1, args['nbr_enc_size']//2)
        #self.explicit_rel_token_bn = nn.BatchNorm1d(args['nbr_enc_size']//2)

        # semantic relationship
        #self.semantic_pos = LearnablePositionalEncoding(d_model=args['node_enc_size'], dropout=0.1, max_len=semantic_pos_max_len)
        self.semantic_pos = LearnablePositionalEncoding2D(32, 100)
        
        self.query_semantic_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        #self.query_semantic_pos = LearnablePositionalEncoding(d_model=args['node_enc_size'], dropout=0.1, max_len=semantic_pos_max_len)
        #self.query_semantic_bn = nn.BatchNorm1d(args['nbr_enc_size'])
        self.key_semantic_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        #self.key_semantic_pos = LearnablePositionalEncoding(d_model=args['node_enc_size'], dropout=0.1, max_len=semantic_pos_max_len)
        #self.key_semantic_bn = nn.BatchNorm1d(args['nbr_enc_size'])
        self.val_semantic_emb = nn.Linear(args['nbr_enc_size'], args['node_enc_size'])
        #self.val_semantic_pos = LearnablePositionalEncoding(d_model=args['node_enc_size'], dropout=0.1, max_len=semantic_pos_max_len)
        #self.val_semantic_bn = nn.BatchNorm1d(args['nbr_enc_size'])
        
        self.semantic_mha = nn.MultiheadAttention(args['nbr_enc_size'], num_heads=4)
        self.semantic_decode = nn.Linear(args['nbr_enc_size'], 1)

        # Target-Neighbor attention
        self.query_v2v_emb = nn.Linear(args['target_agent_enc_size'], args['nbr_enc_size'])
        self.key_v2v_emb = nn.Linear(args['nbr_enc_size'], args['nbr_enc_size'])
        self.val_v2v_emb = nn.Linear(args['nbr_enc_size'], args['nbr_enc_size'])
        self.v2v_attn = nn.MultiheadAttention(args['nbr_enc_size'], num_heads=1)
        self.mix_v2v = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # motif
        #self.spp_enc = ASPP(5, 1)
        #self.spp_agg = SPP
        self.motif_emb = nn.Linear(args['motif_feat_size'], args['motif_enc_size'])
        self.motif_ln = nn.LayerNorm(args['motif_enc_size'])


        # Agent-node attention
        self.query_v2l_emb = nn.Linear(args['node_enc_size'], args['node_enc_size'])
        self.key_v2l_emb = nn.Linear(args['nbr_enc_size']*2, args['node_enc_size'])
        self.val_v2l_emb = nn.Linear(args['nbr_enc_size']*2, args['node_enc_size'])
        self.v2l_attn = nn.MultiheadAttention(args['node_enc_size'], num_heads=1)
        self.mix_v2l = nn.Linear(args['node_enc_size']*2, args['node_enc_size'])

        # Non-linearities
        self.leaky_relu = nn.LeakyReLU()

        # GAT layers
        self.gat = nn.ModuleList([GAT(args['node_enc_size'], args['node_enc_size'])
                                  for _ in range(args['num_gat_layers'])])
        
        print("init done")
        

    def forward(self, inputs, is_test=False, is_val=False):
        """
        Forward pass for PGP encoder
        :param inputs: Dictionary with
            target_agent_representation: torch.Tensor, shape [batch_size, t_h, target_agent_feat_size]
            map_representation: Dict with
                'lane_node_feats': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]
                'lane_node_masks': torch.Tensor, shape [batch_size, max_nodes, max_poses, node_feat_size]

                (Optional)
                's_next': Edge look-up table pointing to destination node from source node
                'edge_type': Look-up table with edge type

            surrounding_agent_representation: Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'vehicle_masks': torch.Tensor, shape [batch_size, max_vehicles, t_h, nbr_feat_size]
                'pedestrians': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
                'pedestrian_masks': torch.Tensor, shape [batch_size, max_peds, t_h, nbr_feat_size]
            agent_node_masks:  Dict with
                'vehicles': torch.Tensor, shape [batch_size, max_nodes, max_vehicles]
                'pedestrians': torch.Tensor, shape [batch_size, max_nodes, max_pedestrians]

            Optionally may also include the following if edges are defined for graph traversal
            'init_node': Initial node in the lane graph based on track history.
            'node_seq_gt': Ground truth node sequence for pre-training

        :return:
        """


        # Encode target agent
        target_agent_feats = inputs['target_agent_representation']  # # [b=32, t=5, feat=5]
        target_agent_embedding = self.leaky_relu(self.target_agent_emb(target_agent_feats)) # [b=32, t=5, feat=16]
        target_agent_embedding = target_agent_embedding.permute(1, 0, 2)
        target_agent_enc = self.target_agent_pos_enc(target_agent_embedding)    # [seq_len, batch_size, embedding_dim]
        target_agent_enc = self.target_agent_tf_enc(target_agent_enc)   # (32, 5, 16)
        target_agent_enc = self.target_agent_tf_norm(target_agent_enc)  # (32, 5, 16)
        target_agent_enc = target_agent_enc.reshape(target_agent_enc.size(0), -1)
        target_agent_enc = self.target_agent_fc_enc(target_agent_enc)   # [b=32, feat=32]


        # Encode lane nodes
        lane_node_feats = inputs['map_representation']['lane_node_feats']   # [32, 164, 20, feat=6]
        lane_node_masks = inputs['map_representation']['lane_node_masks']
        lane_node_embedding = self.leaky_relu(self.node_emb(lane_node_feats))   # [32, 164, 20, feat=16]
        lane_node_enc, _ = self.variable_size_gru_encode(lane_node_embedding, lane_node_masks, self.node_encoder)


        # Encode surrounding agents
        nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']  # # [b=32, max_veh=84, t=5, feat=5]
        nbr_vehicle_feats = torch.cat((nbr_vehicle_feats, torch.zeros_like(nbr_vehicle_feats[:, :, :, 0:1])), dim=-1)   # [b=32, max_veh=84, t=5, feat=6]
        nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks'] # [32, 84, 5, 5]    only 0 or 1
        nbr_vehicle_embedding = self.leaky_relu(self.nbr_emb(nbr_vehicle_feats))
        # nbr_vehicle_embedding = nbr_vehicle_embedding.permute(1, 0, 2)
        # nbr_vehicle_enc, nbr_veh_num = self.variable_size_gru_encode(nbr_vehicle_embedding, nbr_vehicle_masks, self.nbr_enc)
        nbr_veh_enc, raw_nbr_veh_pos, raw_nbr_veh_yaw, nbr_veh_num = self.variable_size_tf_encode(nbr_vehicle_embedding, nbr_vehicle_masks, self.nbr_pos_enc, self.nbr_tf_enc, self.nbr_tf_norm, self.nbr_fc_enc)

        nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        nbr_ped_feats = torch.cat((nbr_ped_feats, torch.ones_like(nbr_ped_feats[:, :, :, 0:1])), dim=-1)
        nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks'] # [32, 77, 5, 5]
        nbr_ped_embedding = self.leaky_relu(self.nbr_emb(nbr_ped_feats))
        # nbr_ped_enc, nbr_ped_num = self.variable_size_gru_encode(nbr_ped_embedding, nbr_ped_masks, self.nbr_enc)
        nbr_ped_enc, raw_nbr_ped_pos, raw_nbr_ped_yaw, nbr_ped_num = self.variable_size_tf_encode(nbr_ped_embedding, nbr_ped_masks, self.nbr_pos_enc, self.nbr_tf_enc, self.nbr_tf_norm, self.nbr_fc_enc)


        nbr_num = nbr_veh_num + nbr_ped_num
        #print("encode nbr done")
        # import pdb; pdb.set_trace()
        # relationship extraction
        # 1. latent relationship
        explicit_rel_all_t, explicit_rel_agg, latent_rel = self.extract_relationship(
            target_agent_feats[:, :, :2],  # ego-veh pos   # (b=32, t=5, f=2)
            raw_nbr_veh_pos,  # nbr_veh pos   # (b+veh, t=5, f=2)
            raw_nbr_ped_pos,  # nbr_ped pos   # (b+veh, t=5, f=2)
            target_agent_feats[:, :, -1],
            raw_nbr_veh_yaw,
            raw_nbr_ped_yaw,
            nbr_veh_num,
            nbr_ped_num,
            target_agent_enc,
            nbr_veh_enc,
            nbr_ped_enc,
        )
        #print("extract rel done")

        fused_rel = self.fuse_relationship(
            explicit_rel_agg, latent_rel, target_agent_enc.device
        )        
        #print("fuse rel done")
        
        new_target_agent_enc, new_nbr_veh_enc, new_nbr_ped_enc = self.agg_relationship(
            fused_rel,
            nbr_veh_num,
            nbr_ped_num,
            target_agent_enc,
            nbr_veh_enc,
            nbr_ped_enc,
            target_agent_enc.device
        )
        
        
        
        # !!!!!
        #target_agent_enc = new_target_agent_enc
        target_agent_enc = torch.cat((target_agent_enc, new_target_agent_enc), dim=1)
        
        nbr_veh_enc = torch.cat((nbr_veh_enc, new_nbr_veh_enc), dim=2)
        nbr_ped_enc = torch.cat((nbr_ped_enc, new_nbr_ped_enc), dim=2)
        nbr_encodings = torch.cat((nbr_veh_enc, nbr_ped_enc), dim=1)    # [32, 84+77, feat=64]
        #nbr_encodings = torch.cat((new_nbr_veh_enc, new_nbr_ped_enc), dim=1)
        # !!!!!
        
        #import pdb; pdb.set_trace()
        
        # (V2V) Target-Neighbor atttention
        # v2v_queries = self.query_v2v_emb(target_agent_enc).unsqueeze(0)     # [1, b=32, 32]
        # v2v_keys = self.key_v2v_emb(nbr_encodings).permute(1, 0, 2)         # [161, b=32, 32]
        # v2v_vals = self.val_v2v_emb(nbr_encodings).permute(1, 0, 2)         # [161, b=32, 32]
        # v2v_att_op, _ = self.v2v_attn(v2v_queries, v2v_keys, v2v_vals)      # [1, b=32, 32]
        # v2v_att_op = v2v_att_op.permute(1, 0, 2)  # [b=32, 1, 32]


        # (V2L) Neighbor-LaneNode attention
        # nbr_hybrid = torch.cat((nbr_encodings, motif_enc), dim=-1)
        v2l_queries = self.query_v2l_emb(lane_node_enc).permute(1, 0, 2)    # [164, b=32, 32]
        v2l_keys = self.key_v2l_emb(nbr_encodings).permute(1, 0, 2)         # [161, b=32, 32]
        # v2l_keys = self.key_v2l_emb(nbr_hybrid).permute(1, 0, 2)         # [161, b=32, 32]
        v2l_vals = self.val_v2l_emb(nbr_encodings).permute(1, 0, 2)         # [161, b=32, 32]
        # v2l_vals = self.val_v2l_emb(nbr_hybrid).permute(1, 0, 2)         # [161, b=32, 32]
        # import pdb; pdb.set_trace()
        v2l_attn_masks = torch.cat((inputs['agent_node_masks']['vehicles'],
                                    inputs['agent_node_masks']['pedestrians']), dim=2)
        v2l_att_op, _ = self.v2l_attn(v2l_queries, v2l_keys, v2l_vals, attn_mask=v2l_attn_masks)
        v2l_att_op = v2l_att_op.permute(1, 0, 2)    # [32, 164, 32]

        # Concatenate with original node encodings and 1x1 conv
        lane_node_enc = self.leaky_relu(self.mix_v2l(torch.cat((lane_node_enc, v2l_att_op), dim=2)))

        # GAT layers
        adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
        for gat_layer in self.gat:
            lane_node_enc += gat_layer(lane_node_enc, adj_mat)

        # Lane node masks
        lane_node_masks = ~lane_node_masks[:, :, :, 0].bool()
        lane_node_masks = lane_node_masks.any(dim=2)
        lane_node_masks = ~lane_node_masks
        lane_node_masks = lane_node_masks.float()

        # Return encodings
        encodings = {
                     'target_agent_encoding': target_agent_enc,
                     # 'target_agent_encoding': v2v_att_op,
                     'context_encoding': {
                                          'v2l_combined': lane_node_enc,
                                          'v2l_combined_masks': lane_node_masks,
                                          'map': None,
                                          'vehicles': None,
                                          'pedestrians': None,
                                          'map_masks': None,
                                          'vehicle_masks': None,
                                          'pedestrian_masks': None
                                          },
                     }

        # Pass on initial nodes and edge structure to aggregator if included in inputs
        if 'init_node' in inputs:
            encodings['init_node'] = inputs['init_node']
            encodings['node_seq_gt'] = inputs['node_seq_gt']
            encodings['s_next'] = inputs['map_representation']['s_next']
            encodings['edge_type'] = inputs['map_representation']['edge_type']

        if is_test:
            return encodings, [explicit_rel_all_t, explicit_rel_agg, latent_rel, fused_rel]
        else:
            return encodings


    def agg_relationship(self,
                         fused_rel,
                         nbr_veh_num,
                         nbr_ped_num,
                         target_agent_enc,
                         nbr_veh_enc,
                         nbr_ped_enc,
                         device
                         ):
                         
        target_agent_enc = target_agent_enc.clone()
        nbr_veh_enc = nbr_veh_enc.clone()
        nbr_ped_enc = nbr_ped_enc.clone()
        
        max_veh_num = nbr_veh_enc.shape[1]
        max_ped_num = nbr_ped_enc.shape[1]

        new_target_agent_enc = []
        new_nbr_veh_enc = []
        new_nbr_ped_enc = []
        
        for bbb in range(len(fused_rel)):
            if nbr_veh_num[bbb]+nbr_ped_num[bbb] == 0:
                new_target_agent_enc.append(target_agent_enc[bbb])
                new_nbr_veh_enc.append(nbr_veh_enc[bbb])
                new_nbr_ped_enc.append(nbr_ped_enc[bbb])
            else:

                assert fused_rel[bbb].size(1) == nbr_veh_num[bbb]+nbr_ped_num[bbb]+1
                one_rel = fused_rel[bbb].squeeze()
                one_target_enc = target_agent_enc[bbb].unsqueeze(0)
                one_nbr_veh_enc = nbr_veh_enc[bbb][:nbr_veh_num[bbb]]
                one_nbr_ped_enc = nbr_ped_enc[bbb][:nbr_ped_num[bbb]]
                all_agent_enc = torch.cat((one_target_enc, one_nbr_veh_enc, one_nbr_ped_enc), dim=0)

                assert all_agent_enc.size(0) == one_rel.size(0)

                agg_features = one_rel.t().mm(all_agent_enc)     # agg by column    # (n, 32)

                #
                one_new_target_agent_enc = agg_features[0]
                new_target_agent_enc.append(one_new_target_agent_enc)

                # split to veh and ped
                agg_veh_fea = agg_features[1:1+nbr_veh_num[bbb]]
                agg_ped_fea = agg_features[1+nbr_veh_num[bbb]:]
                assert agg_veh_fea.size(0) == nbr_veh_num[bbb]
                assert agg_ped_fea.size(0) == nbr_ped_num[bbb]

                # pad zero
                one_new_nbr_veh_enc = torch.zeros((max_veh_num, nbr_veh_enc.shape[2])).to(device)
                one_new_nbr_ped_enc = torch.zeros((max_ped_num, nbr_ped_enc.shape[2])).to(device)
                veh_mask = torch.arange(max_veh_num).unsqueeze(1).expand(-1, nbr_veh_enc.shape[2]) < agg_veh_fea.size(0)
                ped_mask = torch.arange(max_ped_num).unsqueeze(1).expand(-1, nbr_ped_enc.shape[2]) < agg_ped_fea.size(0)
                veh_mask = veh_mask.to(device)
                ped_mask = ped_mask.to(device)
                one_new_nbr_veh_enc = one_new_nbr_veh_enc.masked_scatter(veh_mask, agg_veh_fea)
                one_new_nbr_ped_enc = one_new_nbr_ped_enc.masked_scatter(ped_mask, agg_ped_fea)
                new_nbr_veh_enc.append(one_new_nbr_veh_enc)
                new_nbr_ped_enc.append(one_new_nbr_ped_enc)

                # one_new_nbr_veh_enc


        new_target_agent_enc = torch.stack(new_target_agent_enc)
        new_nbr_veh_enc = torch.stack(new_nbr_veh_enc)
        new_nbr_ped_enc = torch.stack(new_nbr_ped_enc)
        

        return new_target_agent_enc, new_nbr_veh_enc, new_nbr_ped_enc


    def extract_relationship(self,
            ego_pos, nbr_veh_pos, nbr_ped_pos,
            ego_yaw, nbr_veh_yaw, nbr_ped_yaw,
            nbr_veh_num,
            nbr_ped_num,
            target_agent_enc,
            nbr_veh_enc,
            nbr_ped_enc,):

        # nbr_pos
        nbr_pos = torch.cat((nbr_veh_pos, nbr_ped_pos), dim=0)  # (?, 5, 2) # 每个batch里面的邻居数不定
        nbr_pos_cpu = nbr_pos.detach().cpu().numpy()
        # nbr_yaw
        nbr_yaw = torch.cat((nbr_veh_yaw, nbr_ped_yaw), dim=0)  # (?, 5, 2) # 每个batch里面的邻居数不定
        nbr_yaw_cpu = nbr_yaw.detach().cpu().numpy()
        nbr_num = nbr_veh_num + nbr_ped_num

        # ego_pos
        ego_pos_cpu = ego_pos.detach().cpu().numpy()
        # ego_yaw
        ego_yaw_cpu = ego_yaw.detach().cpu().numpy()


        batch_size, traj_time = ego_pos.shape[0], ego_pos.shape[1]

        # explicit relationship
        MMs = []  # Motif的矩阵
        MMs_feas = []  # Motif矩阵对应的特征
        LRs = []    # latent relationship

        cur_nbr_start_idx = 0
        for bbb in range(batch_size):
            tmp_m = []
            tmp_nbr_num = nbr_num[bbb]

            if tmp_nbr_num == 0:
                # 没有邻居，用默认的邻居数
                for ttt in range(traj_time):
                    tmp_m.append(
                        np.zeros(shape=(self.args['motif_default_size'], self.args['motif_default_size']), dtype=np.float32)
                    )
                tmp_m = np.array(tmp_m)  # (5, >=2, >=2)
                tmp_m = torch.from_numpy(tmp_m).to(torch.float32).cuda()
                tmp_m = tmp_m.unsqueeze(0)  # (1, 5, >=2, >=2)
                MMs.append(tmp_m)
                tmp_f = np.zeros(shape=(1, 1, self.args['motif_default_size'], self.args['motif_default_size']), dtype=np.float32)
                tmp_f = torch.from_numpy(tmp_f)
                MMs_feas.append(tmp_f)
            else:

                for ttt in range(traj_time):
                    A = np.zeros(shape=(tmp_nbr_num + 1, tmp_nbr_num + 1), dtype=np.float32)
                    # nbr <-> ego
                    for iii in range(tmp_nbr_num):
                        if np.linalg.norm(
                            ego_pos_cpu[bbb, ttt] -
                            nbr_pos_cpu[cur_nbr_start_idx + iii]
                        ) <= self.args['motif_influence_radius']:
                            
                            yaw_nbr = np.arctan2(np.mean(np.sin(nbr_yaw_cpu[cur_nbr_start_idx+iii])),
                                                 np.mean(np.cos(nbr_yaw_cpu[cur_nbr_start_idx+iii])))
                            yaw_ego = np.arctan2(np.mean(np.sin(ego_yaw_cpu[bbb, ttt])),
                                                 np.mean(np.cos(ego_yaw_cpu[bbb, ttt])))
                            yaw_diff = np.arctan2(np.sin(yaw_nbr-yaw_ego), 
                                                  np.cos(yaw_nbr-yaw_ego))
                            
                            if np.absolute(yaw_diff) <= eval(self.args['yaw_thresh']):
                                A[iii, 0] = 1
                                A[0, iii] = 1

                    # nbr <-> nbr
                    for iii in range(tmp_nbr_num):
                        for jjj in range(iii + 1, tmp_nbr_num):
                            if np.linalg.norm(
                                nbr_pos_cpu[cur_nbr_start_idx + iii] -
                                nbr_pos_cpu[cur_nbr_start_idx + jjj]
                            ) <= self.args['motif_influence_radius']:
                                A[iii+1, jjj+1] = 1
                                A[jjj+1, iii+1] = 1

                                yaw_nbr1 = np.arctan2(np.mean(np.sin(nbr_yaw_cpu[cur_nbr_start_idx+iii])),
                                                      np.mean(np.cos(nbr_yaw_cpu[cur_nbr_start_idx+iii])))
                                yaw_nbr2 = np.arctan2(np.mean(np.sin(nbr_yaw_cpu[cur_nbr_start_idx+jjj])),
                                                      np.mean(np.cos(nbr_yaw_cpu[cur_nbr_start_idx+jjj])))
                                yaw_diff = np.arctan2(np.sin(yaw_nbr1-yaw_nbr2), 
                                                      np.cos(yaw_nbr1-yaw_nbr2))
                                
                                if np.absolute(yaw_diff) <= eval(self.args['yaw_thresh']):
                                    A[iii, 0] = 1
                                    A[0, iii] = 1

                    # ego to first
                    # A = np.roll(A, shift=1, axis=0)  # 将最后一行挪到最前面
                    # A = np.roll(A, shift=1, axis=1)  # 将最后一列挪到最前面
                    if self.args['motif_type'] == 'motif1':
                        M = MotifX(A).M1().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                    
                    elif self.args['motif_type'] == 'motif2':
                        M = MotifX(A).M2().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)

                    elif self.args['motif_type'] == 'motif3':
                        M = MotifX(A).M3().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)

                    elif self.args['motif_type'] == 'motif4':
                        M = MotifX(A).M4().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                    
                    elif self.args['motif_type'] == 'motif5':
                        M = MotifX(A).M5().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                        
                    elif self.args['motif_type'] == 'motif6':
                        M = MotifX(A).M6().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)

                    elif self.args['motif_type'] == 'motif7':
                        M = MotifX(A).M7().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)

                    elif self.args['motif_type'] == 'motif8':
                        M = MotifX(A).M8().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                    
                    elif self.args['motif_type'] == 'motif9':
                        M = MotifX(A).M9().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                    
                    elif self.args['motif_type'] == 'motif10':
                        M = MotifX(A).M10().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                    
                    elif self.args['motif_type'] == 'motif11':
                        M = MotifX(A).M11().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                    
                    elif self.args['motif_type'] == 'motif12':
                        M = MotifX(A).M12().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)

                    elif self.args['motif_type'] == 'motif13':
                        M = MotifX(A).M13().toarray().astype(np.float32)
                        d = M.sum(axis=1)
                        D = np.diag(d)
                        tmp_m.append(M)
                    
                    elif self.args['motif_type'] == 'motif Laplacian':
                        L = D - M
                        tmp_m.append(L)

                    elif self.args['motif_type'] == 'normalized motif Laplacian':
                        L = D - M
                        # print("L:", type(L))
                        # print("D:", type(D))
                        # print("M:", type(M))
                        d = d ** (-0.5)
                        d[d == np.inf] = 0
                        # print("d:", d)
                        normalizedL = np.dot(np.diag(d), np.dot(L, np.diag(d)))
                        # print("normalizedL:", normalizedL)
                        tmp_m.append(normalizedL)

                tmp_m = np.array(tmp_m)  # (5, >=2, >=2)
                tmp_m = torch.from_numpy(tmp_m).to(torch.float32).cuda()
                tmp_m = tmp_m.unsqueeze(0)  # (1, 5, >=2, >=2)
                # print("tmp_m.shape:", tmp_m.shape)
                # tmp_m_fea = SPP(tmp_m, tmp_m.shape[0], tmp_m.shape[-2:], [16, 12, 8, 6, 4, 2])
                # tmp_m_fea = self.spp_enc(tmp_m)  # [1, 5, 24, 24] -> [1, 1, 24, 24]
                # 按最大值归一化
                tmp_m_fea = tmp_m.sum(axis=1).unsqueeze(1)
                if tmp_m_fea.max() != 0:
                    tmp_m_fea = tmp_m_fea / tmp_m_fea.max()
                """
                # 按列求softmax，全0时还都是0
                tmp_m_fea = tmp_m.sum(axis=1).squeeze()
                min_vals = torch.min(tmp_m_fea, dim=0, keepdim=True).values
                max_vals = torch.max(tmp_m_fea, dim=0, keepdim=True).values
                # 如果min和max都为0，则直接返回原始矩阵
                normalized_matrix = torch.where(
                    (min_vals == 0) & (max_vals == 0),
                    tmp_m_fea,
                    (tmp_m_fea - min_vals) / (max_vals - min_vals)
                )
                # 计算每列的指数
                exp_matrix = torch.exp(normalized_matrix)
                # 计算每列的和
                col_sums = torch.sum(exp_matrix, dim=0, keepdim=True)
                # 对每列进行softmax
                tmp_m_fea = exp_matrix / col_sums
                """
                # tmp_m_fea = spp_agg(tmp_m_fea, tmp_m.shape[0], tmp_m.shape[-2:], [16, 12, 8, 6, 4, 2])
                # import pdb; pdb.set_trace()
                # print("tmp_m_fea.shape:", tmp_m_fea.shape)

                # MMs里面按batch存，一个batch里面是一组motif矩阵
                MMs.append(tmp_m)
                MMs_feas.append(tmp_m_fea.clone())

            cur_nbr_start_idx += tmp_nbr_num


            # latent relationship
            one_target_agent_enc = target_agent_enc[bbb].unsqueeze(0).unsqueeze(0)   # (b=1, n=1, f=32)

            if tmp_nbr_num == 0:
                tmp_f = np.zeros(shape=(1, 1, self.args['motif_default_size'], self.args['motif_default_size']), dtype=np.float32)
                tmp_f = torch.from_numpy(tmp_f)
                LRs.append(tmp_f)
            else:
                # one_nbr_encodings = nbr_encodings[bbb][:nbr_num[bbb]].unsqueeze(0) # (b=1, n=?, f=32)
                one_nbr_veh_enc = nbr_veh_enc[bbb][:nbr_veh_num[bbb]].unsqueeze(0)
                one_nbr_ped_enc = nbr_ped_enc[bbb][:nbr_ped_num[bbb]].unsqueeze(0)
                one_nbr_encodings = torch.cat((one_nbr_veh_enc, one_nbr_ped_enc), dim=1)  # [b=1, ?, feat=32]
                assert one_nbr_encodings.size(1) == nbr_num[bbb]
                together = torch.cat((one_target_agent_enc, one_nbr_encodings), dim=1)

                together = self.latent_rel_emb(together)
                together = self.latent_rel_pos(together)

                #
                query = self.query_latent_rel_emb(together)  # [1, n=?, 32]
                #query = self.query_latent_rel_pos(query)
                # query = self.query_latent_rel_bn(query.permute(0, 2, 1)).permute(0, 2, 1)
                # query /= self.args['latent_temperature']
                #print("latent_query:", query.shape)

                key = self.key_latent_rel_emb(together)  # [1, n=?, 32]
                #key = self.key_latent_rel_pos(key)
                # key = self.key_latent_rel_bn(key.permute(0, 2, 1)).permute(0, 2, 1)  # [1, n=?, 32]
                # key /= self.args['latent_temperature']
                #print("latent_key:", key.shape)
                
                attn = (query @ key.transpose(-2, -1)) / math.sqrt(self.args['nbr_enc_size'])  # [1, n=?, n=?]
                #attn = F.softmax(attn, dim=-1)  # 按行求和
                #attn = F.softmax(attn, dim=-2)  # 按列求和

                LRs.append(attn.unsqueeze(1))
                

        return MMs, MMs_feas, LRs


    def fuse_relationship(self, explicit_rel, latent_rel, device):
        assert len(latent_rel) == len(explicit_rel)
        semantic_relationship = []

        for bbb in range(len(latent_rel)):
            latent_rel_token = latent_rel[bbb].to(device).view(1, -1, 1)  # (b, num*num, 1)
            latent_rel_token = self.latent_rel_token_emb(latent_rel_token)  # (b, num*num, 16)
            # latent_rel_token = self.latent_rel_token_bn(latent_rel_token.permute(0, 2, 1)).permute(0, 2, 1)
            latent_rel_token = latent_rel_token.transpose(0, 1)  # (num*num, b, 16)
        

            explicit_rel_token = explicit_rel[bbb].to(device).view(1, -1, 1)  # (b, num*num, 1)
            explicit_rel_token = self.explicit_rel_token_emb(explicit_rel_token)  # (b, num*num, 16)
            # explicit_rel_token = self.explicit_rel_token_bn(explicit_rel_token.permute(0, 2, 1)).permute(0, 2, 1)  # (b, num*num, 16)
            explicit_rel_token = explicit_rel_token.transpose(0, 1)  # (num*num, b, 16)


            # semantic token
            semantic_rel_token = torch.cat((latent_rel_token, explicit_rel_token), dim=2)   # (num*num, b, 32)
            nn, b, d = semantic_rel_token.size()
            #import pdb; pdb.set_trace()
            # [2d pos]
            semantic_rel_token = semantic_rel_token.view(b, d, int(np.sqrt(nn)), int(np.sqrt(nn)))
            #print("1semantic_rel_token:", semantic_rel_token.shape)
            #import pdb; pdb.set_trace()
            semantic_rel_token = self.semantic_pos(semantic_rel_token)
            semantic_rel_token = semantic_rel_token.view(nn, b, d)
            
            
            #print("b semantic_rel_token:", semantic_rel_token.shape, semantic_rel_token.min(), semantic_rel_token.max())
            #semantic_rel_token = self.semantic_pos(semantic_rel_token)
            #print("a semantic_rel_token:", semantic_rel_token.shape, semantic_rel_token.min(), semantic_rel_token.max())
            #print("semantic_rel_token:", semantic_rel_token.shape)
            semantic_rel_query = self.query_semantic_emb(semantic_rel_token)
            #semantic_rel_query = self.query_semantic_pos(semantic_rel_query)
            # semantic_rel_query = self.query_semantic_bn(semantic_rel_query.permute(1, 2, 0)).permute(2, 0, 1)
            #semantic_rel_query = semantic_rel_query.transpose(0, 1)
            #print("semantic_rel_query:", semantic_rel_query.shape, semantic_rel_query.min(), semantic_rel_query.max())
            
            semantic_rel_key = self.key_semantic_emb(semantic_rel_token)
            #semantic_rel_key = self.key_semantic_pos(semantic_rel_key)
            # semantic_rel_key = self.key_semantic_bn(semantic_rel_key.permute(1, 2, 0)).permute(2, 0, 1)
            #semantic_rel_key = semantic_rel_key.transpose(0, 1)
            
            semantic_rel_value = self.val_semantic_emb(semantic_rel_token)
            #semantic_rel_value = self.val_semantic_pos(semantic_rel_value)
            # semantic_rel_value = self.val_semantic_bn(semantic_rel_value.permute(1, 2, 0)).permute(2, 0, 1)
            #semantic_rel_value = semantic_rel_value.transpose(0, 1)
            
            #
            semantic_rel_token_new, _  = self.semantic_mha(semantic_rel_query, semantic_rel_key, semantic_rel_value)    # (num*num, b=1, 32)
            
                
                
            semantic_rel_token_new = semantic_rel_token_new.transpose(0, 1)  # (b=1, num*num, 32)
            semantic_rel_token_new = self.semantic_decode(semantic_rel_token_new)   # (b=1, num*num, 1)
            
            sss = int(np.sqrt(semantic_rel_token_new.size(1)))
            semantic_rel_token_new = semantic_rel_token_new.view(1, sss, sss)
            

            semantic_rel_token_new /= self.args['fuse_temperature']

            # semantic_rel_token_new = F.softmax(semantic_rel_token_new, dim=-2)  # 按列求和
            semantic_rel_token_new = F.softmax(semantic_rel_token_new, dim=-1)  # 按行求和
            semantic_relationship.append(semantic_rel_token_new.clone())

        return semantic_relationship




    @staticmethod
    def variable_size_tf_encode(feat_embedding, masks, pos_enc, tf_enc, nbr_tf_norm, fc_enc):
        """
        Returns GRU encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        # mask shape: [32, 84, 5, 5]
        # masks[:, :, :, 0] shape: [32, 84, 5]
        masks_for_batching = ~masks[:, :, :, 0].bool()
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])  # (number of all batches, 5, 16)
        raw_pos = feat_embedding_batched[:, :, :2]
        raw_yaw = feat_embedding_batched[:, :, -1]

        nbr_num = masks_for_batching.sum(axis=1).view(-1, 1)  # num in each batch

        # Pack padded sequences
        seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
        seq_lens_batched = seq_lens[seq_lens != 0].cpu()
        if len(seq_lens_batched) != 0:
            feat_embedding_batched = feat_embedding_batched.permute(1, 0, 2)    # (5, b, 16)
            feat_embedding_batched = pos_enc(feat_embedding_batched)  # [seq_len, batch_size, embedding_dim]
            feat_embedding_batched = tf_enc(feat_embedding_batched)  # (b, 5, 16)
            feat_embedding_batched = nbr_tf_norm(feat_embedding_batched)  # [b=number in all batches, feat=32]
            feat_enc_batched = feat_embedding_batched.reshape(feat_embedding_batched.size(0), -1)
            feat_enc_batched = fc_enc(feat_enc_batched)  # [b=number in all batches, feat=32]

            # Scatter back to appropriate batch index
            masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, feat_enc_batched.shape[-1])
            encoding = torch.zeros(masks_for_scattering.shape, device=device)
            encoding = encoding.masked_scatter(masks_for_scattering, feat_enc_batched)  # (32, 84, 5)
        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            # import pdb; pdb.set_trace()
            hidden_state_size = fc_enc.out_features
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

        return encoding, raw_pos, raw_yaw, nbr_num

    @staticmethod
    def variable_size_gru_encode(feat_embedding: torch.Tensor, masks: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
        """
        Returns GRU encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3]) # (number, 5, 16)
        nbr_num = masks_for_batching.sum(axis=1).squeeze()

        # Pack padded sequences
        seq_lens = torch.sum(1 - masks[:, :, :, 0], dim=-1)
        seq_lens_batched = seq_lens[seq_lens != 0].cpu()
        if len(seq_lens_batched) != 0:
            feat_embedding_packed = pack_padded_sequence(feat_embedding_batched, seq_lens_batched,
                                                         batch_first=True, enforce_sorted=False)

            # Encode
            _, encoding_batched = gru(feat_embedding_packed)
            encoding_batched = encoding_batched.squeeze(0)

            # Scatter back to appropriate batch index
            masks_for_scattering = masks_for_batching.squeeze(3).repeat(1, 1, encoding_batched.shape[-1])
            encoding = torch.zeros(masks_for_scattering.shape, device=device)
            encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            hidden_state_size = gru.hidden_size
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

        return encoding, nbr_num

    @staticmethod
    def build_adj_mat(s_next, edge_type):
        """
        Builds adjacency matrix for GAT layers.
        """
        batch_size = s_next.shape[0]
        max_nodes = s_next.shape[1]
        max_edges = s_next.shape[2]
        adj_mat = torch.diag(torch.ones(max_nodes, device=device)).unsqueeze(0).repeat(batch_size, 1, 1).bool()

        dummy_vals = torch.arange(max_nodes, device=device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        dummy_vals = dummy_vals.float()
        s_next[edge_type == 0] = dummy_vals[edge_type == 0]
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_edges)
        src_indices = torch.arange(max_nodes).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        adj_mat[batch_indices[:, :, :-1], src_indices[:, :, :-1], s_next[:, :, :-1].long()] = True
        adj_mat = adj_mat | torch.transpose(adj_mat, 1, 2)

        return adj_mat


class GAT(nn.Module):
    """
    GAT layer for aggregating local context at each lane node. Uses scaled dot product attention using pytorch's
    multihead attention module.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize GAT layer.
        :param in_channels: size of node encodings
        :param out_channels: size of aggregated node encodings
        """
        super().__init__()
        self.query_emb = nn.Linear(in_channels, out_channels)
        self.key_emb = nn.Linear(in_channels, out_channels)
        self.val_emb = nn.Linear(in_channels, out_channels)
        self.att = nn.MultiheadAttention(out_channels, 1)

    def forward(self, node_encodings, adj_mat):
        """
        Forward pass for GAT layer
        :param node_encodings: Tensor of node encodings, shape [batch_size, max_nodes, node_enc_size]
        :param adj_mat: Bool tensor, adjacency matrix for edges, shape [batch_size, max_nodes, max_nodes]
        :return:
        """
        queries = self.query_emb(node_encodings.permute(1, 0, 2))
        keys = self.key_emb(node_encodings.permute(1, 0, 2))
        vals = self.val_emb(node_encodings.permute(1, 0, 2))
        att_op, _ = self.att(queries, keys, vals, attn_mask=~adj_mat)

        return att_op.permute(1, 0, 2)
