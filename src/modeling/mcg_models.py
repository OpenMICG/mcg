import copy

import numpy as np
import torch
import torch.nn.functional as F
from horovod import torch as hvd
from src.modeling.timesformer.vit import TimeSformer

from src.modeling.med import (BertEmbeddings, BertEncoder, BertLMPredictionHead, BertModel, BertPooler, BertLMHeadModel,
                              BertPreTrainedModel)
from src.utils.basic_utils import load_json, load_jsonl, save_frames_grid
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange


class McgBaseModel(nn.Module):
    def __init__(self, config=None, input_format='RGB', video_enc_cfg=None, temp=0.07):
        super().__init__()

        # learnable temperature parameter
        self.temp = nn.Parameter(torch.ones([]) * temp)

        # visual encoder
        visual_model_cls = eval(video_enc_cfg['cls'])
        self.visual_encoder = visual_model_cls(model_cfg=video_enc_cfg,
                                               input_format=input_format,
                                               cross_attention_config=config)

        # bert configuration
        self.bert_config = config
        text_width = self.bert_config.hidden_size
        self.itc_token_type = self.bert_config.itc_token_type

        # projection layer
        self.vision_proj = nn.Linear(768, 256)
        self.text_proj = nn.Linear(text_width, 256)


    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)


class McgForPretrain(McgBaseModel):
    def __init__(self, config, video_enc_cfg, input_format='RGB'):
        super(McgForPretrain, self).__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg)

        # text encoder
        self.text_encoder = BertModel(config=self.bert_config, add_pooling_layer=False)
        self.use_mask_prob = 0

    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        visual_inputs = batch['visual_inputs']

        device = visual_inputs.device
        b, t, c, h, w = visual_inputs.shape

        video_embeds = self._forward_visual_embeds(visual_inputs)

        # we compute normalized feats for unmasked visual inputs only, used for ITC
        video_feat = F.normalize(self.vision_proj(video_embeds[:, 0, :]), dim=-1)
        video_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(device)

        # text embeddings and features
        text_embeds, sents, word_attn, text_feat = self._forward_text_feats(batch)

        # ========== VTC loss ==========
        gathered_video_feats = hvd.allgather(video_feat)
        gathered_text_feats = hvd.allgather(text_feat)

        assert self.itc_token_type == 'cls', 'Support CLS tokens for ITC only, find {}.'.format(self.itc_token_type)
        sim_v2t = video_feat @ gathered_text_feats.t() / self.temp
        sim_t2v = text_feat @ gathered_video_feats.t() / self.temp

        sim_targets = torch.zeros_like(sim_v2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.eye(b)

        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * sim_targets, dim=1).mean()

        vtc_loss = (loss_v2t + loss_t2v) / 2

        # ========= MLM loss ==========
        text_atts = batch['text_input_mask']
        if 'mlm_labels' in batch:
            mlm_labels = batch['mlm_labels']
            mlm_text_input_ids = batch['mlm_text_input_ids']
            mlm_loss, mlm_logits, mlm_labels = self.compute_mlm(input_ids=mlm_text_input_ids,
                                                                text_input_mask=text_atts,
                                                                video_embeds=video_embeds,
                                                                video_atts=video_atts,
                                                                mlm_labels=mlm_labels
                                                                )
        else:
            mlm_logits = mlm_loss = mlm_labels = None

        # ========= MCL loss ==========
        patch_emb = F.normalize(self.vision_proj(video_embeds[:, 1:, :]), dim=-1)
        word_emb = F.normalize(self.text_proj(text_embeds[:, 1:, :]), dim=-1)
        mcl_loss = self.compute_mcl(batch, patch_emb, word_emb, word_attn, sents, mcl_local_temperature = 0.1)

        return dict(
            itc_loss=vtc_loss,
            mlm_scores=mlm_logits,  # (B, Lt, vocab_size),  only text part
            mlm_loss=mlm_loss,  # (BxLt)
            mlm_labels=mlm_labels,  # (1, Lt), with -100 indicates ignored positions
            mcl_loss=mcl_loss
        )

    def _forward_visual_embeds(self, visual_inputs):
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # image features
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)

        return video_embeds

    def _forward_text_feats(self, batch):
        text_output = self.text_encoder(batch['text_input_ids'],
                                        batch['text_input_mask'],
                                        batch['token_type_ids'],
                                        return_dict=True,
                                        mode='text')
        last_layer_attn = text_output.attentions[-1][:, :, 0, 1:].mean(dim=1)
        text_embeds = text_output.last_hidden_state  # b, Lt, fsz=768
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        text_embeds, sents, last_atten_pt = self.text_encoder.encoder.aggregate_tokens(text_embeds.unsqueeze(1),
                                                                                       batch['text_input_ids'],
                                                                                       last_layer_attn)
        last_atten_pt = last_atten_pt[:, 1:]
        text_embeds = text_embeds[:, 0]
        return text_embeds, sents, last_atten_pt, text_feat

    def compute_mlm(self, input_ids, text_input_mask, video_embeds, video_atts, mlm_labels):
        # forward text features with masked_input_ids
        text_output = self.text_encoder(input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                             )
        text_embeds = text_output.last_hidden_state

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, video_atts], dim=1)
        embedding_output = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs = self.text_encoder(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                 )

        txt_len = text_input_mask.shape[1]
        txt_output = encoder_outputs.last_hidden_state[:, :txt_len]

        mlm_logits = self.text_encoder.cls(txt_output)

        loss_fct = CrossEntropyLoss()
        mlm_loss = loss_fct(mlm_logits.view(-1, self.bert_config.vocab_size), mlm_labels.view(-1))

        return mlm_loss, mlm_logits, mlm_labels

    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)

        # [NOTE] BERT is initialized from huggingface pre-trained weights. 
        # if bert_weights_path:
        #     load_multimodal_encoder_state_dict_with_mismatch(self.cross_encoder, bert_weights_path)
        #     load_mlm_head_state_dict_with_mismatch(self.mlm_head, bert_weights_path)

    def compute_mcl(self, batch, patch_emb, word_emb, word_attn, sents, mcl_local_temperature):
        bz = patch_emb.size(0)
        mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(batch["visual_inputs"]).bool()
        word_emb = torch.squeeze(word_emb, 1)

        # ========= loss_word =========
        atten_sim = torch.bmm(word_emb, patch_emb.permute(0, 2, 1))
        atten_scores = F.softmax(atten_sim / mcl_local_temperature, dim=-1)  # bz, 196, 111
        word_atten_output = torch.bmm(atten_scores, patch_emb)
        word_atten_output = F.normalize(word_atten_output, dim=-1)
        with torch.no_grad():
            atten_weights = word_attn.detach()
            word_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)

        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)
        word_sim = torch.bmm(word_emb, word_atten_output.permute(0, 2, 1)) / mcl_local_temperature
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(word_emb).long().repeat(bz)
        loss_word_1 = torch.sum(F.cross_entropy(word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz
        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz
        loss_word = (loss_word_1 + loss_word_2) / 2.

        # ========= loss_patch =========
        atten_sim = torch.bmm(patch_emb, word_emb.permute(0, 2, 1))
        patch_num = patch_emb.size(1)
        atten_sim[mask.unsqueeze(1).repeat(1, patch_num, 1)] = float("-inf")
        atten_scores = F.softmax(atten_sim / mcl_local_temperature, dim=-1)  # bz, 196, 111
        patch_atten_output = torch.bmm(atten_scores, word_emb)
        with torch.no_grad():
            img_attn_map = self.visual_encoder.model.blocks[-1].attn.attention_map.detach()
            atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
            patch_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                atten_weight = atten_weight.clip(torch.quantile(atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                patch_atten_weights.append(atten_weight.clone())
            patch_atten_weights = torch.stack(patch_atten_weights)

        patch_atten_weights /= patch_atten_weights.sum(dim=1, keepdims=True)
        patch_sim = torch.bmm(patch_emb, patch_atten_output.permute(0, 2, 1)) / mcl_local_temperature
        patch_num = patch_sim.size(1)
        patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(patch_num).type_as(patch_emb).long().repeat(bz)
        loss_patch_1 = torch.sum(F.cross_entropy(patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz
        patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
        loss_patch_2 = torch.sum(F.cross_entropy(patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz
        loss_patch = (loss_patch_1 + loss_patch_2) / 2.

        # ========= loss_local =========
        loss_local = loss_patch + loss_word
        return loss_local


class McgForOpenEnded(McgBaseModel):
    def __init__(self, config, video_enc_cfg, tokenizer, input_format='RGB'):
        super(McgForOpenEnded, self).__init__(config, video_enc_cfg=video_enc_cfg)
        self.task_type = config.task
        self.config = config
        self.text_encoder = BertModel(config=self.bert_config, add_pooling_layer=False)
        self.text_decoder = BertLMHeadModel(config=self.bert_config)
        self.tokenizer = tokenizer

    def forward(self, batch, is_train=True):
        visual_inputs = batch['visual_inputs']
        device = visual_inputs.device

        # forward visual
        visual_inputs = visual_inputs.transpose(1, 2)  # timeSformer asks for (b, c, t, h, w) as input.
        image_embeds = self.visual_encoder.forward_features(visual_inputs, return_all_tokens=True)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        # forward text
        question_input_mask = batch['text_input_mask']
        question_input_ids = batch['text_input_ids']
        question_output = self.text_encoder(question_input_ids,
                                            attention_mask=question_input_mask,
                                            return_dict=True,
                                            mode='text')
        text_embeds = question_output.last_hidden_state

        # fusion
        attention_mask = torch.cat([question_input_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)
        question_output = self.text_encoder(encoder_embeds=embedding_output,
                                            attention_mask=attention_mask,
                                            return_dict=True,
                                            mode='fusion')

        predictions = []
        if is_train:
            ans_input_ids = batch['ans_input_ids']
            ans_input_mask = batch['ans_input_mask']
            answer_targets = batch['answer_targets']
            answer_output = self.text_decoder(ans_input_ids,
                                              attention_mask=ans_input_mask,
                                              encoder_hidden_states=question_output.last_hidden_state,
                                              labels=answer_targets,
                                              return_dict=True,
                                              reduction='none')
            loss = answer_output.loss
            loss = loss.sum() / visual_inputs.shape[0]
        else:
            num_beams = 3
            question_states = question_output.last_hidden_state.repeat_interleave(num_beams, dim=0)
            question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long, device=question_states.device)
            model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask": question_atts}

            bos_ids = torch.full((visual_inputs.shape[0], 1),
                                 fill_value=self.tokenizer.bos_token_id,
                                 device=visual_inputs.device)

            outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                 max_length=10,
                                                 min_length=1,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 **model_kwargs)
            # encoder_attention_mask
            for output in outputs:
                prediction = self.tokenizer.decode(output, skip_special_tokens=True)
                predictions.append(prediction)

            loss = 0
        return dict(loss=loss, logits=predictions)

