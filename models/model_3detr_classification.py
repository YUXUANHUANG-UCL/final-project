# Copyright (c) Facebook, Inc. and its affiliates.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
import torch.nn.functional as F
from models.helpers import GenericMLP, GenericMLP_class
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)

def build_preencoder(args):
    mlp_dims = [256, 512, args.enc_dim]  # Adjusted MLP dimensions for XYZ-only input
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder



def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder = TransformerDecoder(
        input_dim=3,
        dec_dim = args.dec_dim, 
        nhead=args.dec_nhead,
        num_layers=args.dec_nlayers,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    return decoder


class ClassificationModel(nn.Module):
    def __init__(self, pre_encoder, encoder, decoder, use_enc_only, pretrain, mlp_dropout=0.3, num_classes=55, encoder_dim=256, decoder_dim=256, position_embedding="fourier", num_queries=256):
        super().__init__()
        hidden_dims = [encoder_dim*4, encoder_dim]  # Customize the hidden layer dimensions of the MLP
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.decoder = decoder
        self.num_queries = num_queries
        self.mlp_dropout = mlp_dropout
        self.num_classes = num_classes
        self.use_enc_only = use_enc_only
        self.pretrain = pretrain
        '''
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=False,
            output_use_activation=False,
            hidden_use_bias=False,
        )
        self.mlp_class = GenericMLP_class(
            #input_dim=2048,  # Update input dimension to 3 for (x, y, z) coordinates
            input_dim=3072,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=False,
            dropout=mlp_dropout,
            hidden_use_bias=False,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False
        )
        
        self.mlp = GenericMLP(
            #input_dim=2048,  # Update input dimension to 3 for (x, y, z) coordinates
            input_dim=1024,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=False,
            dropout=mlp_dropout,
            hidden_use_bias=False,
            output_use_bias=True,
            output_use_activation=False,
            output_use_norm=False
        )
        '''
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    '''
    def get_classification_predictions(self, encoder_outputs):
        # Reshape the decoder outputs
        batch_size, num_points, _ = encoder_outputs.shape
        output_reshaped = encoder_outputs.view(batch_size, -1)  # [batch_size, num_points * 3]
        # Pass the decoder outputs through the MLP classifier
        classification_output = self.mlp_class(output_reshaped)

        return classification_output
    '''
    def forward(self, x):
        # pre_encoder_xyz (B, npoints, 3)
        pre_encoder_xyz, pre_encoder_features, _ = self.pre_encoder(x)  # Ignore the returned indices
        # Transpose the features tensor
        # pre_encoder_features ( npoint, B, mlp[-1])
        pre_encoder_features = pre_encoder_features.permute(2, 0, 1)
        if not self.pretrain:
            encoder_output, enc_features, enc_inds = self.encoder(pre_encoder_features, xyz=pre_encoder_xyz)  # Ignore the returned indices
            enc_features_softmax = F.softmax(enc_features, dim=-1)
            max_values, max_indices = torch.topk(enc_features_softmax, k=3, dim=-1)
            max_values = max_values.permute(1, 0, 2)
            encoder_output = encoder_output + 0.01*max_values
        else:
            encoder_output = pre_encoder_xyz
        if self.use_enc_only:
            x = F.relu(self.bn1(self.conv1(encoder_output.permute(0, 2, 1))))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.dropout(self.fc2(x))))
            x = self.fc3(x)
            x = F.log_softmax(x, dim=1)
            return x
            #return self.get_classification_predictions(encoder_output)
        print('use dedcoder')
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(0, 2, 1)
        ).permute(0, 2, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3
        decoder_outputs = self.decoder(pre_encoder_xyz, encoder_output, enc_features)
        '''
        query_xyz, query_embed = self.get_query_embeddings(encoder_output.contiguous())
        enc_pos = self.pos_embedding(encoder_output)
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(1, 0, 2)
        tgt = query_xyz.permute(2, 0, 1)
        decoder_outputs, _ = self.decoder(tgt, enc_features, query_pos=query_embed, pos=enc_pos)
        '''
        outputs = self.get_classification_predictions(decoder_outputs)
        return outputs


def build_3detr(args):
    preencoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)  # Add decoder layer
    model = ClassificationModel(preencoder, encoder, decoder, args.use_enc_only, args.pretrain, args.mlp_dropout, args.num_classes, args.enc_dim, args.dec_dim, args.pos_embed, args.nqueries)
    return model

