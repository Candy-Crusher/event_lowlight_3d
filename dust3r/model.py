# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class with ControlNet-inspired event voxel control
# --------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn
import os
from packaging import version
import huggingface_hub
import math

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed
from third_party.raft import load_RAFT
from dust3r.lightup_net import EvLightEnhancer

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

# 添加CrossAttention类定义
class CrossAttention(nn.Module):
    """多头交叉注意力模块，用于event特征对image特征的注意力"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., rope=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
    def forward(self, query, key, value, query_pos=None, key_pos=None):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        
        # 投影查询、键和值
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # 应用位置编码（如果有）
        if self.rope is not None and query_pos is not None and key_pos is not None:
            # 检查rope对象的接口
            if hasattr(self.rope, 'apply_rotary'):
                q = self.rope.apply_rotary(q, positions=query_pos)
                k = self.rope.apply_rotary(k, positions=key_pos)
            else:
                # 如果没有apply_rotary方法，则不应用位置编码
                pass
            
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders with event voxel control.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry). Event voxel data is used as a control signal.
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 use_event_control=False,  # Flag to enable event control
                 event_in_channels=5,    # Event voxel input channels
                 use_lowlight_enhancer=False,  # Flag to enable EvLightEnhancer
                 use_cross_attention_for_event=False,  # Flag to enable cross attention in event encoder
                 event_enhance_mode='none',  # 'none', 'easy', or 'complex'
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.use_event_control = use_event_control
        self.event_in_channels = event_in_channels
        self.use_lowlight_enhancer = use_lowlight_enhancer
        self.use_cross_attention_for_event = use_cross_attention_for_event
        self.event_enhance_mode = event_enhance_mode
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        # Event control initialization
        if self.use_event_control:
            self._init_event_control(croco_kwargs.get('img_size', 224), croco_kwargs.get('patch_size', 16),
                                    croco_kwargs.get('enc_embed_dim', 768), croco_kwargs.get('dec_embed_dim', 768))
        
        # Low-light enhancer initialization
        if self.use_lowlight_enhancer:
            print("Initializing EvLightEnhancer for low-light image enhancement")
            print(f"Event input channels: {event_in_channels}")
            print(f"Enhancement mode: {self.event_enhance_mode}")
            self.enhancer = EvLightEnhancer(
                mode=self.event_enhance_mode,
                image_channels=3, 
                event_channels=event_in_channels
            )

    def zero_module(self, module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module

    def _init_event_control(self, img_size, patch_size, enc_embed_dim, dec_embed_dim):
        """ Initialize modules for processing event voxel data and injecting it into the network. """
        # embedding
        # self.event_embed = nn.Sequential(
        #     nn.Conv2d(self.event_in_channels, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, enc_embed_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(enc_embed_dim, enc_embed_dim, kernel_size=patch_size, stride=patch_size)  # Downsample to patch size
        # )
        self.event_embed = deepcopy(self.patch_embed)
        self.event_embed.proj = nn.Conv2d(self.event_in_channels, enc_embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Zero-convolution层
        self.enc_zero_conv_in = self.zero_module(nn.Conv1d(enc_embed_dim, enc_embed_dim, 1))
        self.enc_zero_conv_out = self.zero_module(nn.Conv1d(enc_embed_dim, enc_embed_dim, 1))
        # Trainable copy
        self.enc_blocks_trainable = nn.ModuleList([deepcopy(blk) for blk in self.enc_blocks])
        # set to trainable
        for blk in self.enc_blocks_trainable:
            for param in blk.parameters():
                param.requires_grad = True
        
        # 如果启用交叉注意力
        if self.use_cross_attention_for_event:
            # 从croco_kwargs获取num_heads，但减少头数以节省显存
            num_heads = min(8, self.croco_args.get('enc_num_heads', 16))  # 最多使用8个头
            
            # 只在部分层使用交叉注意力，每6层使用一次
            attn_layers = list(range(0, len(self.enc_blocks), 6))
            if len(self.enc_blocks) - 1 not in attn_layers:
                attn_layers.append(len(self.enc_blocks) - 1)  # 确保最后一层有交叉注意力
            
            # 为event encoder添加稀疏的交叉注意力模块
            self.event_cross_attns = nn.ModuleDict({
                str(i): CrossAttention(
                    dim=enc_embed_dim,
                    num_heads=num_heads,  # 减少头数
                    qkv_bias=True,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    rope=self.rope if hasattr(self, 'rope') else None
                ) for i in attn_layers
            })
            
            # 归一化层
            self.event_norms = nn.ModuleDict({
                str(i): nn.LayerNorm(enc_embed_dim) for i in attn_layers
            })
            
            self.img_norms = nn.ModuleDict({
                str(i): nn.LayerNorm(enc_embed_dim) for i in attn_layers
            })
            
            # 交叉注意力权重 - 初始化为0，逐渐学习
            self.cross_attn_weights = nn.ParameterDict({
                str(i): nn.Parameter(torch.zeros(1)) for i in attn_layers
            })
            
            # 初始化为小权重，确保训练初期不影响原有模型
            for i in attn_layers:
                for p in self.event_cross_attns[str(i)].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p, gain=0.01)
                    else:
                        nn.init.zeros_(p)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
            'decoder': [self.dec_blocks, self.dec_blocks2],
            'all':     [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2,
                        self.downstream_head1, self.downstream_head2]
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape, event_voxel=None):
        # 初始化SNR map为None
        snr_map = None
        
        # 如果启用了低光增强器并且提供了事件数据，先进行低光增强处理
        if self.use_lowlight_enhancer and event_voxel is not None:
            # Ensure everything is on the same device
            if image.device != event_voxel.device:
                event_voxel = event_voxel.to(image.device)
            
            # 使用EvLightEnhancer增强图像，并获取SNR map
            image, snr_map = self.enhancer(image, event_voxel)  # 保存SNR map以便后续使用

        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B, N, C = x.size()
        posvis = pos
        
        # 保存原始image特征副本（如果需要SNR融合）
        if self.use_event_control and event_voxel is not None and snr_map is not None:
            x_original = x.clone()

        # Process event voxel if provided
        event_features = None
        if self.use_event_control and event_voxel is not None:
            # Event voxel shape: [B, 6, H, W]
            c, c_pos = self.event_embed(event_voxel, true_shape=true_shape)
            
            # 保持原有的方式：通过zero convolution输入
            c_in = self.enc_zero_conv_in(c.transpose(1, 2)).transpose(1, 2) + x  # [B, N, enc_embed_dim]
            
            # 依次处理每个encoder层
            for i, blk in enumerate(self.enc_blocks):
                # image encoder保持不变
                x = blk(x, posvis)
                
                # 修改event encoder，添加交叉注意力机制
                if self.use_cross_attention_for_event and str(i) in self.event_cross_attns:
                    # 归一化特征
                    c_norm = self.event_norms[str(i)](c_in)
                    x_norm = self.img_norms[str(i)](x)
                    
                    # 计算注意力权重
                    weight = torch.sigmoid(self.cross_attn_weights[str(i)])
                    
                    # 应用交叉注意力：event特征关注image特征
                    cross_attn_output = self.event_cross_attns[str(i)](
                        c_norm,    # 查询：event特征
                        x_norm,    # 键：image特征  
                        x_norm,    # 值：image特征
                        c_pos,     # 查询位置
                        posvis     # 键值位置
                    )
                    
                    # 残差连接：将交叉注意力结果添加到event特征
                    c_in = c_in + weight * cross_attn_output
                    
                    # 然后通过原来的event encoder block
                    c_in = self.enc_blocks_trainable[i](c_in, posvis)
                else:
                    # 使用原有的event encoder处理
                    c_in = self.enc_blocks_trainable[i](c_in, posvis)
                
                # 在最后一层，根据SNR mask来融合event特征和image特征
                if i == len(self.enc_blocks) - 1:
                    c_out = self.enc_zero_conv_out(c_in.transpose(1, 2)).transpose(1, 2)
                    
                    # 如果有SNR map，使用它来加权融合
                    if snr_map is not None:
                        # 将1通道的SNR map扩展为3通道
                        snr_map_3ch = snr_map.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                        
                        # 使用与image相同的patch_embed处理SNR map
                        snr_patches, _ = self.patch_embed(snr_map_3ch, true_shape=true_shape)
                        
                        # 取第一个通道作为权重（因为三个通道的值是相同的）
                        snr_weight = snr_patches[:, :, 0:1]
                        
                        # 归一化SNR值到[0,1]范围
                        snr_weight = torch.sigmoid(snr_weight)
                        
                        # 进行加权融合
                        try:
                            # SNR高的地方用image encoder结果，SNR低的地方用event encoder结果
                            x = x * snr_weight + c_out * (1 - snr_weight)
                        except RuntimeError as e:
                            # 保持原有特征不变
                            pass
                    else:
                        # 如果没有SNR map，使用原来的方式
                        try:
                            x = x + c_out
                        except RuntimeError as e:
                            pass

        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, event_voxel1=None, event_voxel2=None):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(
                torch.cat((img1, img2), dim=0),
                torch.cat((true_shape1, true_shape2), dim=0),
                torch.cat((event_voxel1, event_voxel2), dim=0) if event_voxel1 is not None else None
            )
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, event_voxel1)
            out2, pos2, _ = self._encode_image(img2, true_shape2, event_voxel2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        event_voxel1 = view1.get('event_voxel')
        event_voxel2 = view2.get('event_voxel')
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2],
                event_voxel1[::2] if event_voxel1 is not None else None,
                event_voxel2[::2] if event_voxel2 is not None else None
            )
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(
                img1, img2, shape1, shape2, event_voxel1, event_voxel2
            )

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
