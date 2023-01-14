from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import BertTokenizer
import jieba

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


# change model
# with / without prefix
# MODEL_PATH = "/mnt/2-7new-try/task_augment/pretrain_models/mt5-small"
# MODEL_PATH = "/mnt/2-7new-try/task_augment/codes/new_tests/offical_t5_2_mt5/contradiction_res/checkpoint-12000"
# MODEL_PATH = "/mnt/2-7new-try/task_augment/codes/new_tests/offical_t5_2_mt5/contradiction_res"
# MODEL_PATH = "/mnt/2-7new-try/task_augment/codes/new_tests/offical_t5_2_mt5/contradiction_res_pe"
# MODEL_PATH = "/mnt/2-7new-try/task_augment/codes/new_tests/offical_t5_2_mt5/entailment_res_pe"
MODEL_PATH = "/mnt/2-7new-try/task_augment/codes/new_tests/offical_t5_2_mt5/neutral_res_pe"

# tokenizer = MT5Tokenizer.from_pretrained(MODEL_PATH)
tokenizer = T5PegasusTokenizer.from_pretrained(MODEL_PATH)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATH)


task_prefix = "contradiction: "
# task_prefix = ""
input_sequence_1 = "因为咱们请来了一个真正的专家,李玫瑾老师,每次观众都知道李老师一来,大的案件就发生了"
input_sequence_2 = "我我后来又给你写了一封信啊"
input_sequence_3 = "被告人王某1在包头市XX区办事处九郡世纪广场东单元22楼东户室内，盗窃放在门口的被害人王某2、王某3的手包各一个，被屋内人员发现后报警，"
input_sequence_4 = "包头市九原区人民检察院指控，王某偷了孙某200元，但王某拒不承认"
# 转为list
input_sequences = [input_sequence_1]
input_sequences.extend([input_sequence_2, input_sequence_3, input_sequence_4])


input_ids = tokenizer([task_prefix + sequence for sequence in input_sequences],
                   padding=True,
                   return_tensors="pt").input_ids
outputs = model.generate(input_ids)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)


outputs = model.generate(input_ids,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 eos_token_id=tokenizer.sep_token_id,
                                 top_k=3,
                                 max_length=300)
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(result)
result = ''.join(tokenizer.batch_decode(outputs, skip_special_tokens=True)).replace(' ', '')
print(result)


