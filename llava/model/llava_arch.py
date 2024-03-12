#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from tome.merge import *
import json

def draw_cross_attention(o_input_ids,o_input_embeds,image_embeds):
    from transformers import AutoTokenizer
    from torch.nn.functional import cosine_similarity
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    model_path = "/home/lzh/llx/models/models--liuhaotian--llava-v1.5-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    input_ids = o_input_ids[2:-5]
    input_embeds = o_input_embeds[2:-5]
    similarity = cosine_similarity(input_embeds.unsqueeze(1), image_embeds.unsqueeze(0), dim=2)
    text_image_avg_similarity = torch.mean(similarity, dim=0).cpu().numpy()
    similarity_matrix = text_image_avg_similarity.reshape(24, 24)

    image_to_add = Image.open('/home/lzh/llx/LLaVA/images/scienceqa/ant.png')
    image_to_add = image_to_add.resize((336, 336))  # 调整图片大小

    # 画每一个token对应的相似度图

    # num_text_tokens = similarity.size(0)
    # for i in range(num_text_tokens):
    #     # 取出第 i 个 text token 与所有 image tokens 的相似度
    #     text_image_similarity = similarity[i].cpu().numpy()  # 转换为 NumPy 数组
    #     # 重塑为 24x24 矩阵
    #     similarity_matrix = text_image_similarity.reshape(24, 24)
        
    #     # 创建一个新的子图
    #     plt.figure(figsize=(4, 4))
    #     # 绘制热力图
    #     plt.imshow(similarity_matrix, cmap='viridis')
    #     plt.colorbar()  # 显示颜色条
    #     plt.title(f'Similarity Heatmap for Text {tokenizer.decode(input_ids[i])}')
    #     plt.xlabel('Image Token X-coordinate')
    #     plt.ylabel('Image Token Y-coordinate')
    #     plt.savefig(f'/home/zhihang/LLaVA/images/dog_cat/{tokenizer.decode(input_ids[i])}.png', dpi=300)

    # 整个文本和图片的相似度图
    plt.figure(figsize=(12, 6))
    # 绘制热力图
    plt.subplot(1, 2, 1)
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()  # 显示颜色条
    plt.title(f'Text: {tokenizer.decode(input_ids)}')
    plt.xlabel('Image Token X-coordinate')
    plt.ylabel('Image Token Y-coordinate')

    #绘制图片
    plt.subplot(1, 2, 2)
    plt.imshow(image_to_add)
    # plt.title('dog')
    plt.axis('off')  # 不显示坐标轴

    #保存
    plt.savefig('/home/lzh/llx/LLaVA/images/scienceqa/ant-ant,.jpg', dpi=300)
def draw_protect_token(tensor1,x=0):
    import numpy as np
    import matplotlib.pyplot as plt
    tensor = tensor1.reshape((24,24)).cpu().numpy()
    plt.imshow(tensor, cmap='gray')
    plt.colorbar()
    if x==0:
        plt.savefig('/home/lzh/llx/protect_toke_before', dpi=300)
    else:
        plt.savefig('/home/lzh/llx/protect_toke_after', dpi=300)
    # for i in range(5,10):
    # a = torch.where(tensor1 > 2, torch.tensor(1), torch.tensor(0))
    # tensor = a.reshape((24,24)).cpu().numpy()
    # plt.imshow(tensor, cmap='gray')
    # plt.savefig(f'/home/lzh/llx/protect_token1.png', dpi=300)

def calculate_entropy(tensor,tempreature):
    # First convert tensor values to probabilities using softmax function
    tensor_prob = torch.nn.functional.softmax(tensor/tempreature, dim=-1)
    # Compute entropy
    entropy = -torch.sum(tensor_prob * torch.log(tensor_prob+1e-9), dim=-1)
    return entropy

def save_similarity(o_input_ids,o_input_embeds,image_embeds):
    from torch.nn.functional import cosine_similarity
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from .. import config
    topnum = config.args.topnum
    o_input_ids = o_input_ids[37:-16]
    o_input_embeds = o_input_embeds[37:-16]
    model_path = "/home/lzh/llx/models/models--liuhaotian--llava-v1.5-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    dic = {}
    for key, value in zip(o_input_ids, o_input_embeds):
        if key.item() not in dic and len(tokenizer.decode(key.item()))>1 and tokenizer.decode(key.item())!="Context":
            dic[key.item()] = value
    input_ids = torch.tensor(list(dic.keys()))
    input_embeds = torch.stack(list(dic.values()))
    # 计算每个文本token和图像token相似度
    similarity = cosine_similarity(input_embeds.unsqueeze(1).to(image_embeds.device), image_embeds.unsqueeze(0), dim=2)
    # 计算每个文本token的熵
    input_entropy = calculate_entropy(similarity,0.01)
    min_val = torch.min(similarity)
    similarity = similarity - min_val
    similarity /= input_entropy.unsqueeze(1)
    input_similarity_sum = similarity.sum(dim=0)
    torch.save(input_similarity_sum,"/home/lzh/llx/attn_map/similarity.pt")



def cross_attention(o_input_ids,o_input_embeds,image_embeds,idx):
    from torch.nn.functional import cosine_similarity
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from .. import config

    topnum = config.args.topnum
    o_input_ids = o_input_ids[37:-5]
    o_input_embeds = o_input_embeds[37:-5]
    # o_input_ids = o_input_ids[37:-16]
    # o_input_embeds = o_input_embeds[37:-16]
    model_path = "/home/lzh/llx/models/models--liuhaotian--llava-v1.5-13b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # print_info = {
    #     'problem_idx':idx,
    #     'problem':tokenizer.decode(o_input_ids),
    #     'token_chosed': []
    # }
    # print(idx)
    # print(tokenizer.decode(o_input_ids))
    # 去重操作，保证input_ids和input_embeds位置一样，比较方便查看选到哪些token
    dic = {}
    for key, value in zip(o_input_ids, o_input_embeds):
        if key.item() not in dic and len(tokenizer.decode(key.item()))>1 and tokenizer.decode(key.item())!="Context":
            dic[key.item()] = value
    input_ids = torch.tensor(list(dic.keys()))
    input_embeds = torch.stack(list(dic.values()))
    # 计算每个文本token和图像token相似度
    similarity = cosine_similarity(input_embeds.unsqueeze(1).to(image_embeds.device), image_embeds.unsqueeze(0), dim=2)
    # 计算每个文本token的熵
    input_entropy = calculate_entropy(similarity,0.01)
    min_val = torch.min(similarity)
    similarity = similarity - min_val
    similarity /= input_entropy.unsqueeze(1)
    input_similarity_sum = similarity.sum(dim=0)

    # input_entropy[torch.where(input_similarity_sum < 0)[0]] = 100
    # values, indices = torch.topk(-input_entropy, min(input_entropy.shape[0],retain_num))
    # # values, indices = torch.topk(-input_entropy, retain_num)
    # protect_text_similarity = similarity[indices,:] # 取熵最小的五个值
    # protect_token_idx = torch.where(input_entropy < 5)[0]
    # for i in indices:
    #     print(tokenizer.decode(input_ids[i]),input_entropy[i].item(), end=' ')
    #     print_info['token_chosed'].append({
    #                 'token': tokenizer.decode(input_ids[i]),
    #                 'entropy': input_entropy[i].item()
    #             })
    # with open('/home/lzh/llx/vqa_output.json', 'a', encoding='utf-8') as f:
    #     json.dump(print_info, f, ensure_ascii=False, indent=4)
    #     f.write('\n')
    # if mode == "vote":
    #     protect_image_token = (protect_text_similarity > sim_thresh).sum(dim=0).to(protect_text_similarity.device)
    #     # draw_protect_token(protect_image_token,2)
    #     protect_image_token_idx = torch.where(protect_image_token>2)[0]
    # elif mode == "sum":
    # mask0 = protect_text_similarity > sim_thresh
    # mask_values = mask0 * protect_text_similarity
    # protect_image_token = mask_values.sum(dim=0)
    # print(torch.count_nonzero(protect_image_token))
    values, protect_image_token_idx = torch.topk(input_similarity_sum, topnum)

    mask = torch.ones(input_similarity_sum.size(0), dtype=bool)
    mask[protect_image_token_idx] = False
    return image_embeds[protect_image_token_idx],image_embeds[mask]
    
def avg_pooling(image_token_idx):
    pass

def token_merging(image_embeds,r,size):
    merge,_ = bipartite_soft_matching(image_embeds,r,False,False)
    image_embeds_merged,sized = merge_wavg(merge,image_embeds,size)
    return image_embeds_merged,sized

def fixnum_token(img_token,max_token,r):
    token_len = img_token.shape[1]
    size = None
    while(1):
        if(token_len <= max_token):
            break
        img_token,size = token_merging(img_token,min(r,token_len-max_token),size)
        token_len -= min(r,token_len-max_token)
    return img_token,token_len

def dual_token_merging(cur_input_ids,cur_input_embeds,cur_image_features,ratio,max_pro,idx):
    protect_img_token,normal_img_token = cross_attention(cur_input_ids,cur_input_embeds,cur_image_features,idx)
    # print("protect_img_token.shape",protect_img_token.shape)
    protect_img_token = protect_img_token.unsqueeze(0)
    normal_img_token = normal_img_token.unsqueeze(0)
    target_num = int((protect_img_token.shape[1]+normal_img_token.shape[1])*ratio)
    # merge protect_img_token
    protect_img_token,lenpro = fixnum_token(protect_img_token,max_pro,8)
    # merge normal_img_token
    max_nor = target_num-lenpro
    normal_img_token,lennor = fixnum_token(normal_img_token,max_nor,32)
    # print("after_len:",protect_img_token.shape[1]+normal_img_token.shape[1],"pro:",protect_img_token.shape[1])
    return torch.cat((protect_img_token.squeeze(0),normal_img_token.squeeze(0)),dim=0)

def normal_token_merging(img_token,ratio):
    img_token = img_token.unsqueeze(0)
    maxnum = int(img_token.shape[1]*ratio)
    from .. import config
    topnum = config.args.topnum
    img_token,_ = fixnum_token(img_token,topnum,32)
    img_token,_ = fixnum_token(img_token,maxnum,8)
    return img_token.squeeze(0)


def downsample_features(features, original_height, original_width, downsample_size):
    # 确定变形后的形状(N, H, W, C)，准备提取左上角的值
    features = features.reshape((original_height, original_width, features.shape[-1]))
    # 下采样后的新高度和宽度
    new_height, new_width = original_height // downsample_size, original_width // downsample_size
    # 初始化下采样后的新特征
    downsampled = torch.zeros((new_height, new_width, features.shape[-1]), dtype=features.dtype)
    # 选择左上角的值，步进为下采样的大小
    for i in range(new_height):
        for j in range(new_width):
            downsampled[i, j, :] = features[i * downsample_size, j * downsample_size, :]
    # 改变形状以匹配目标输出大小 (N, H*W, C)
    return downsampled.reshape((new_height * new_width, features.shape[-1]))

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None]
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                print("the problem idx:",self.model.idx)
                from .. import config
                ratio = config.args.ratio
                max_pro = config.args.max_pro
                use_merge = config.args.use_merge
                if use_merge == True:
                    if self.model.flag == 1:
                        self.model.flag += 1
                        # cur_image_features = dual_token_merging(cur_input_ids,cur_input_embeds_1,cur_image_features,ratio,max_pro,self.model.idx)
                        cur_image_features = normal_token_merging(cur_image_features,ratio)
                        print("cur_image_features.shape",cur_image_features.shape)
                        self.merge_image = cur_image_features
                    else:
                        cur_image_features = self.merge_image
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_input_id = torch.cat(cur_input_ids_noim)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    from .. import config
                    ratio = config.args.ratio
                    max_pro = config.args.max_pro
                    merge_mode = config.args.merge_mode
                    if merge_mode == "downsample":
                        cur_image_features = downsample_features(cur_image_features, 24, 24, 2)
                    elif merge_mode == "tome":
                        cur_image_features = normal_token_merging(cur_image_features,ratio)
                    elif merge_mode == "mytome":
                        import time
                        if self.model.flag == 1 and cur_image_idx <= len(self.merge_image):
                            cur_image_features = dual_token_merging(cur_input_id,cur_input_embeds,cur_image_features,ratio,max_pro,self.model.idx)
                            self.merge_image.append(cur_image_features)
                        else:
                            cur_image_features = self.merge_image[cur_image_idx]
                    save_similarity(cur_input_id,cur_input_embeds,cur_image_features)
                    print("the problem idx:",self.model.idx)
                    print("merge_mode",merge_mode)
                    print("cur_image_features.shape",cur_image_features.shape)
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            self.model.flag = 2
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
