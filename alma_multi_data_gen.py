import os
from datetime import timedelta
from tqdm import tqdm
import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object

result_path = './results/'
model_path = 'yanolja/EEVE-Korean-10.8B-v1.0'
lora_path = 'qwopqwop/ALMA-EEVE-v1'

os.makedirs(result_path, exist_ok=True)
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))  # 24 hours
accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    quantization_config=bnb_config, 
    trust_remote_code=True,
    device_map={"": accelerator.process_index},
)
model.config.use_cache = False
model = PeftModel.from_pretrained(model, lora_path)
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model.eval()

data_mt = joblib.load('./data_mt.pkl')
prompts_all = []
for i in range(len(data_mt['ko'])):
    if os.path.exists(f'{result_path}{i}.pkl'):
        print(f"Skipping {index}")
    else:
        prompts_all.append([i, data_mt['en'][i],data_mt['ko'][i]])

accelerator.wait_for_everyone()
with accelerator.split_between_processes(prompts_all) as prompts:
    for data in tqdm(prompts):
        index, en_text, ko_text = data
        if os.path.exists(f'{result_path}{index}.pkl'):
            print(f"Skipping {index}")
            continue

        en_prompt = f"Translate this from English to Korean:\nEnglish: {en_text}\nKorean:"
        ko_prompt = f"Translate this from Korean to English:\nKorean: {ko_text}\nEnglish:"
        
        prompt_tokenized= tokenizer([en_prompt,ko_prompt], return_tensors="pt", padding=True, max_length=1024, truncation=True).to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, num_beams=5, max_new_tokens=1024)
        output_tokenized=output_tokenized[:, len(prompt_tokenized["input_ids"][0]):]
        output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)
        joblib.dump(
            {
                "index": index,
                "en_text":en_text,
                "ko_text": ko_text,
                "en_ko_outputs": output[0],
                "ko_en_outputs": output[1]
            },
        f'{result_path}{index}.pkl'
        )
