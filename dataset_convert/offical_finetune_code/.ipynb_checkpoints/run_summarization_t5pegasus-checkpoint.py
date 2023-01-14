#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import time
import datatime
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import jieba
import torch
from datasets import load_dataset, load_metric
from tools import progressbar
from pynvml import *

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version

"""-------- OCNLI + t5-pegasus-small ------注意output_dir可能要改--"""###------------------------一定先看看有没有冻结层
"""  contradiction_res_pe
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-small
--source_prefix "contradiction: "
--train_file ../data/contradiction_train.50k.json
--validation_file ../data/contradiction_dev.json
--output_dir ./contradiction_pe_ft67f/
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--do_train
--text_column sentence1 
--summary_column sentence2 
--overwrite_output_dir
--predict_with_generate
--do_eval
****###------------------------一定先看看有没有冻结层
python run_summarization_t5pegasus.py \
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-small \
--source_prefix "contradiction: " \
--train_file ../data/contradiction_train.50k.json \
--validation_file ../data/contradiction_dev.json \
--output_dir ./contradiction_pe_ft67f/ \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--do_train \
--text_column sentence1 \
--summary_column sentence2 \
--overwrite_output_dir \
--predict_with_generate \
--do_eval
"""

"""  entailment_res_pe ###------------------------一定先看看有没有冻结层
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-small
--do_train
--do_eval
--source_prefix "entailment: "
--train_file ../data/entailment_train.50k.json
--validation_file ../data/entailment_dev.json
--text_column sentence1 
--summary_column sentence2 
--output_dir ./entailment_res_pe/
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--overwrite_output_dir
--predict_with_generate
"""

"""  neutral_res_pe  ###------------------------一定先看看有没有冻结层
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-small
--do_train
--do_eval
--source_prefix "neutral: "
--train_file ../data/neutral_train.50k.json
--validation_file ../data/neutral_dev.json
--text_column sentence1 
--summary_column sentence2 
--output_dir ./neutral_res_pe/
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--overwrite_output_dir
--predict_with_generate
"""

"""-------- ocnli-public + t5-pegasus-base ------注意output_dir可能要改--"""###------------------------一定先看看有没有冻结层
"""  contradiction_public
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-base
--source_prefix "contradiction: "
--train_file ../data/ocnli-public/contradiction_train.json
--validation_file ../data/ocnli-public/contradiction_dev.json
--output_dir ./contradiction_public/
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--do_train
--do_eval
--text_column sentence1 
--summary_column sentence2 
--overwrite_output_dir
--predict_with_generate
********* ###------------------------一定先看看有没有冻结层
python run_summarization_t5pegasus.py \
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-base \
--source_prefix "contradiction: " \
--train_file ../data/ocnli-public/contradiction_train.json \
--validation_file ../data/ocnli-public/contradiction_dev.json \
--output_dir ./contradiction_public/ \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--do_train \
--text_column sentence1 \
--summary_column sentence2 \
--overwrite_output_dir \
--predict_with_generate \
--do_eval
"""

"""  entailment_public ###------------------------一定先看看有没有冻结层
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-base
--do_train
--do_eval
--source_prefix "entailment: "
--train_file ../data/ocnli-public/entailment_train.json
--validation_file ../data/ocnli-public/entailment_dev.json
--text_column sentence1 
--summary_column sentence2 
--output_dir ./entailment_public/
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--overwrite_output_dir
--predict_with_generate
******* ###------------------------一定先看看有没有冻结层
python run_summarization_t5pegasus.py \
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-base \
--source_prefix "entailment: " \
--train_file ../data/ocnli-public/entailment_train.json \
--validation_file ../data/ocnli-public/entailment_dev.json \
--output_dir ./entailment_public/ \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--do_train \
--text_column sentence1 \
--summary_column sentence2 \
--overwrite_output_dir \
--predict_with_generate \
--do_eval
"""

"""  neutral_res_pe     ---------------------一定先看看有没有冻结层
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-base
--do_train
--do_eval
--source_prefix "neutral: "
--train_file ../data/ocnli-public/neutral_train.json
--validation_file ../data/ocnli-public/neutral_dev.json
--text_column sentence1 
--summary_column sentence2 
--output_dir ./neutral_public/
--per_device_train_batch_size=4
--per_device_eval_batch_size=4
--overwrite_output_dir
--predict_with_generate
******* ###------------------------一定先看看有没有冻结层
python run_summarization_t5pegasus.py \
--model_name_or_path /mnt/3_new-try/task_augment/pretrain_models/t5-pegasus-base \
--source_prefix "neutral: " \
--train_file ../data/ocnli-public/neutral_train.json \
--validation_file ../data/ocnli-public/neutral_dev.json \
--output_dir ./neutral_public/ \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--do_train \
--text_column sentence1 \
--summary_column sentence2 \
--overwrite_output_dir \
--predict_with_generate \
--do_eval
"""


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.19.0.dev0")   # 这个有点碍事，我删掉了

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")  # https://blog.csdn.net/m0_45478865/article/details/108752417
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:  # 模型参数
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:  # 数据训练参数  # data_args.
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(  # 不知道是不是这里应该变为true
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        # default=128,
        default=256,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
                    "Useful for multilingual models like mBART where the first generated token"
                    "needs to be the target language token (Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:  # 如果全空
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:  # 不是全空
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

# 这个是t5pegasus的分词器
class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


summarization_name_mapping = {  # 数据集&key对
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def my_logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open("log/generate_text-to-text_model", 'a+') as f_log:
            f_log.write(s + '\n')


def main(data_args=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))  # 感觉上面相当于通过注解和类名重写了一些子类，里面重新定义了一些东西
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging 日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:  每个过程的日志
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")  # 此处会输出

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:  # true、false、not true
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model. 初始化模型之前设置种子
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    # 可以自己提供csv/json数据文件，或者提供hub上的公开数据集的名字
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    # 对于csv/json，脚本会使用第一列作为full text，第二列座位摘要（除非你为***来指定列名）
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:  # 如果写了数据集名称，就去下载
        # Downloading and loading a dataset from the hub.
        # https://huggingface.co/docs/datasets/loading
        # raw_datasets = load_dataset(   # 原始代码
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     cache_dir=model_args.cache_dir,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
        raw_datasets = datasets.load_from_disk("/mnt/2-7new-try/task_augment/datasets/cnn_dailymail")
        print(raw_datasets)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(  # Model_config T5Config
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # tokenizer = AutoTokenizer.from_pretrained(  # remove
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     use_fast=model_args.use_fast_tokenizer,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    tokenizer = T5PegasusTokenizer.from_pretrained(model_args.model_name_or_path)  # new add
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):  # false
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:  # false
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length):  # false
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""  # summarize:

    # Preprocessing the datasets.  # 准备数据集
    # We need to tokenize inputs and targets.  # 对输入和目标进行编码
    if training_args.do_train:  # true
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):  # false
        assert (
                data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.   获取列名
    ## 数据集k-v对 获得到("article","highlights")
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]  # article
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]  # highlights
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training. 设置最大目标长度
    max_target_length = data_args.max_target_length  # 128
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):  # false
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None  # 丢弃不全的pair
        inputs, targets = [], []
        for i in range(len(examples[text_column])):  # 1000
            if examples[text_column][i] is not None and examples[summary_column][i] is not None:  # 如果两边数据都不缺，就将其放入inputs/targets中
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]  # 每一个句子前面都加上"summarize: "
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)  # 编码  data暂时只有input_ids和attention_mask（居然都是1）

        # Setup the tokenizer for targets  我没看出来哪里不一样
        with tokenizer.as_target_tokenizer():  # 为了给模型准备好摘要的targets，我们使用as_target_tokenizer来控制targets所对应的特殊token
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:  # false
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]  # 在data里面增加了'labels'标签，现在是3个了、
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:  # false None
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        with training_args.main_process_first(desc="train dataset map pre-processing"):  # 在preprocess_function中对每个batch进行编码
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,  # None
                remove_columns=column_names,  # ['article','highlights','id']
                load_from_cache_file=not data_args.overwrite_cache,  # not false
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:  # 评估
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(  # 数据收集器data collator，将我们处理好的输入喂给模型
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("./cache/rouge.py")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer # 训练器Trainer类：主要用于指定使用的模型，数据，微调过程所用参数的信息。
    trainer = Seq2SeqTrainer(  # 类中包含用于训练，验证，预测的方法：trainer.train(train_dataset)，trainer.evaluate(eval_dataset)，trainer.predict(test_dataset)。
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # new add
    def print_gpu_utilization():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")
    def print_summary(result):
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        print_gpu_utilization()
    print("开始训练之前")
    print_gpu_utilization()

    # Training
    if training_args.do_train:
        # new add
        # for name, param in model.named_parameters():
        #     if param.requires_grad:  # 输出需要梯度的层
        #         print(name, param.size())
        # # unfreeze_layers = ['block.0.', 'block.1.', 'block.2.']
        # # unfreeze_layers = ['block.3.','block.4.','block.5.','block.6.', 'block.7.', 'final_layer_norm']
        # unfreeze_layers = ['block.6.', 'block.7.', 'final_layer_norm']
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
        #     for ele in unfreeze_layers:
        #         if ele in name:
        #             param.requires_grad = True
        #             break
        # # 验证一下
        # for name, param in model.named_parameters():
        #     if param.requires_grad:  # 输出需要梯度的层
        #         print(name, param.size())
        # # end

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:  # false
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:  # false
            checkpoint = last_checkpoint

        # with torch.no_grad():  # 不行，因为训练需要计算题都
        #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if hasattr(torch.cuda, 'empty_cache'):  # 不行，大概是因为我电脑显存实在太小
            torch.cuda.empty_cache()
        train_start_time=time.time()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        train_end_time=time.time()
        my_logging("Train " + training_args.output_dir + "use " + model_args.model_name_or_path + " cost" + str(train_end_time-train_start_time)+"s\n")
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
    my_logging("\n\n------------------New Generation----------------"+time.asctime()+"\n")
