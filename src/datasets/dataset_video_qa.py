import os
import random
import numpy as np
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER, load_file
from src.utils.metrics import get_wups
from src.datasets.dataset_base import McgBaseDataset
from src.datasets.randaugment import TemporalConsistentRandomAugment
from pywsd.utils import lemmatize_sentence


stopwords = load_file('/path/to/your/data/project/mcg/src/datasets/stopwords.txt')


class McgVideoQADataset(McgBaseDataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """
    open_ended_qa_names = ["frameqa", "msrvtt_qa", "msvd_qa", "next_qa_oe"]

    def __init__(self, task_type, datalist, tokenizer, img_lmdb_dir,
                 fps=3, num_frm=3, frm_sampling_strategy="rand",
                 max_img_size=1000, max_txt_len=20, vocab=None,
                 ensemble_n_clips=1, return_label=True, is_train=False, random_sample_clips=True,
                 video_fmt='.mp4', img_db_type='lmdb'):
        super(McgVideoQADataset, self).__init__(
            datalist, tokenizer, img_lmdb_dir, img_db_type=img_db_type,
            fps=fps, num_frm=num_frm,
            frm_sampling_strategy=frm_sampling_strategy,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.ensemble_n_clips = ensemble_n_clips
        self.return_label = return_label
        self.is_train = is_train
        self.task_type = task_type
        self.random_sample_clips = random_sample_clips
        if task_type not in ['next_qa', 'next_qa_oe']:
            self.idx2data = {d["question_id"]: d for group in datalist for d in group[1]}
        else:
            self.idx2data = {d["index"]: d for group in datalist for d in group[1]}
        self.vocab = vocab

        self.video_fmt = video_fmt

        if self.is_train:
            self.randaug = TemporalConsistentRandomAugment(N=2, M=5,
                                                           augs=['Identity', 'Contrast', 'Brightness', 'Sharpness',
                                                                 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
                                                                 'Rotate', 'HorizontalFlip'])
        else:
            self.randaug = None

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 5
        for _ in range(num_retries):
            vid_id, examples = self.datalist[index]  # one video with multiple examples
            if self.ensemble_n_clips > 1:
                raise NotImplementedError('Do not support multiple clips for now.')
            else:
                video_path = os.path.join(self.img_db_dir, vid_id + self.video_fmt)
                vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size,
                                                                  width=self.max_img_size)

            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            if self.randaug:
                vid_frm_array = self.randaug(vid_frm_array.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            examples = [self._get_single_example(e) for e in examples]

            return dict(
                vid=vid_frm_array,
                examples=examples,
                n_examples=len(examples),  # used to create image feature copies.
                vid_id=vid_id
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"],
        )
        if self.task_type != 'next_qa_oe':
            example["index"] = data["question_id"]
        else:
            example["index"] = data["index"]

        if not self.return_label:
            example["label"] = None

        return example

    def evaluate_qa(self, results, ref_file_add=None):
        multi_ref_ans = False
        if os.path.exists(ref_file_add):
            add_ref = load_file(ref_file_add)
            multi_ref_ans = True
        else:
            add_ref = {}

        preds = []
        gts = []
        answer_types = []
        answer_types_over = []
        vids = []
        qids = []
        answer_type2idx = dict(
            frameqa={"object": 0, "number": 1, "color": 2, "location": 3},
            msrvtt_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
            msvd_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
            next_qa_oe={k: idx for idx, k in enumerate(["CH", "CW", "DB", "DC", "DL", "DO", "TC", "TN", "TP"])},
            next_qa_over={k: idx for idx, k in enumerate(["C", "T", "D"])}
        )
        if self.task_type == 'next_qa_oe':
            for r in results:
                preds.append(r['answer'])
                gts.append(r['data']['answer'])
                answer_types.append(r['data']['answer_type'])
                vids.append(str(r['data']['vid_id'].split('/')[-1]))
                qids.append(str(r['question_id']))
        else:
            qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
            for qid, pred_ans in qid2pred_ans.items():
                preds.append(pred_ans)
                gt_data = self.idx2data[qid]
                gt_ans = gt_data["answer"]
                if self.task_type in self.open_ended_qa_names:
                    answer_types.append(answer_type2idx[self.task_type][gt_data["answer_type"]])
                gts.append(gt_ans)

        preds = np.array(preds)
        gts = np.array(gts)
        answer_types = np.array(answer_types)
        answer_types_over = np.array(answer_types_over)
        metrics = dict()

        # for open-ended wups
        if self.task_type == 'next_qa_oe':
            qids = np.array(qids)
            vids = np.array(vids)
            ref_num = 0
            num = {ans_type: 0 for ans_type, ans_type_idx in answer_type2idx[self.task_type].items()}
            over_num = {'C': 0, 'T': 0, 'D': 0}
            metrics["overall_wups_0"] = 0
            metrics["overall_wups_9"] = 0
            for item in over_num:
                metrics[f"{item}_wups_0"] = 0
                metrics[f"{item}_wups_9"] = 0
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                metrics[f"{ans_type}_wups_0"] = 0
                metrics[f"{ans_type}_wups_9"] = 0

            # pred: prediction of one example
            # gt: target of one example
            # answer_types: question type of one example
            # vid: video id
            # qid: question id of one video
            for pred, gt, answer_type, vid, qid in zip(preds, gts, answer_types, vids, qids):
                ref_num += 1
                num[answer_type] += 1
                over_num[answer_type[0]] += 1
                pred_ans = remove_stop(pred)
                gt_ans = remove_stop(gt)
                if multi_ref_ans and (vid in add_ref):
                    add_ref_ans = remove_stop(add_ref[vid][qid])
                    if answer_type not in ['DB', 'DC']:
                        wups_0 = max(float(get_wups(pred_ans, gt_ans, 0)),
                                     float(get_wups(pred_ans, add_ref_ans, 0)))
                        wups_9 = max(float(get_wups(pred_ans, gt_ans, 0.9)),
                                     float(get_wups(pred_ans, add_ref_ans, 0.9)))
                    else:
                        wups_0 = 1 if pred_ans == gt_ans or pred_ans == add_ref_ans else 0
                        wups_9 = wups_0
                else:
                    if answer_type not in ['DB', 'DC']:
                        wups_0 = float(get_wups(pred_ans, gt_ans, 0))
                        wups_9 = float(get_wups(pred_ans, gt_ans, 9))
                    else:
                        wups_0 = 1 if pred_ans == gt_ans else 0
                        wups_9 = wups_0

                metrics[f"{answer_type}_wups_0"] += wups_0
                metrics[f"{answer_type}_wups_9"] += wups_9
                metrics[f"{answer_type[0]}_wups_0"] += wups_0
                metrics[f"{answer_type[0]}_wups_9"] += wups_9
                metrics["overall_wups_0"] += wups_0
                metrics["overall_wups_9"] += wups_9

            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                if num[ans_type] != 0:
                    metrics[f"{ans_type}_wups_0"] = metrics[f"{ans_type}_wups_0"] / num[ans_type] * 100
                    metrics[f"{ans_type}_wups_9"] = metrics[f"{ans_type}_wups_9"] / num[ans_type] * 100
                else:
                    metrics[f"{ans_type}_wups_0"] = 0
                    metrics[f"{ans_type}_wups_9"] = 0

            # over type wups
            for k, v in over_num.items():
                if over_num[k] != 0:
                    metrics[f"{k}_wups_0"] = metrics[f"{k}_wups_0"] / v * 100
                    metrics[f"{k}_wups_9"] = metrics[f"{k}_wups_9"] / v * 100
                else:
                    metrics[f"{k}_wups_0"] = 0
                    metrics[f"{k}_wups_9"] = 0

            # over_all wups
            metrics["overall_wups_0"] = metrics["overall_wups_0"] / ref_num * 100
            metrics["overall_wups_9"] = metrics["overall_wups_9"] / ref_num * 100
        # for open-ended accuracy
        else:
            metrics["overall_acc"] = float(np.mean(preds == gts))
            ratios = dict()
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                answer_type_mask = answer_types == ans_type_idx
                answer_type_corrects = (preds[answer_type_mask] == gts[answer_type_mask])
                metrics[f"{ans_type}_acc"] = float(np.mean(answer_type_corrects)) if len(
                    answer_type_corrects) != 0 else 0
                ratios[f"{ans_type}_ratio"] = [1. * len(answer_type_corrects) / len(answer_types),
                                               len(answer_type_corrects)]
            metrics["ratios"] = ratios

        return metrics


# To collate data
class VideoQACollator(object):
    def __init__(self, tokenizer, max_length=20, task_type="action", n_options=5, is_train=False, patch_size=16, cfg=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.n_options = n_options
        self.istrain = is_train
        self.patch_size = patch_size
        if cfg != None:
            self.cfg = cfg

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W) (2, 16, 3, 224, 224)

        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])

        ans_input_ids = None
        ans_input_mask = None
        answer_str_list = None
        labels = None
        answer_targets = None

        question_ids = [d["question_id"] for d in text_examples]
        indexes = [d["index"] for d in text_examples]

        text_str_list = [d["q_str"] for d in text_examples]  # (B, )
        answer_str_list = [d["label"] for d in text_examples]

        # encode question
        batch_enc = self.tokenizer.batch_encode_plus(
            text_str_list,
            max_length=self.max_length,
            padding='max_length',  # 40
            return_tensors="pt",
            truncation=True
        )

        text_input_ids = batch_enc.input_ids  # (B, L)
        text_input_mask = batch_enc.attention_mask  # (B, L)

        # if self.task_type == "next_qa_oe":
        # cls_token ---> enc_token
        batch_enc.input_ids[:, 0] = self.tokenizer.enc_token_id

        if self.istrain:
            batch_enc_ans = self.tokenizer.batch_encode_plus(
                answer_str_list,
                max_length=64,
                padding='max_length',
                return_tensors="pt",
                truncation=True
            )
            ans_input_ids = batch_enc_ans.input_ids  # (B, L)
            ans_input_mask = batch_enc_ans.attention_mask  # (B, L)

            # cls_token_id ---> bos_token_id
            ans_input_ids[:, 0] = self.tokenizer.bos_token_id
            answer_targets = ans_input_ids.masked_fill(ans_input_ids == self.tokenizer.pad_token_id, -100)
        return dict(
            visual_inputs=visual_inputs,  # (B, #frm, H, W, C)
            text_input_ids=text_input_ids,
            text_input_mask=text_input_mask,
            ans_input_ids=ans_input_ids,
            ans_input_mask=ans_input_mask,
            answer_targets=answer_targets,
            question_ids=question_ids,
            labels=labels,
            indexes=indexes
        )


def remove_stop(sentence):
    words = lemmatize_sentence(sentence)
    words = [w for w in words if not w in stopwords]
    return ' '.join(words)
