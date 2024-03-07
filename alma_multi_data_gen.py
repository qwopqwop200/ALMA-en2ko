import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

model_path = 'yanolja/EEVE-Korean-10.8B-v1.0'
lora_path = './adapter_model-379/'

accelerator = Accelerator()
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
    prompts_all.append([data_mt['en'][i],data_mt['ko'][i]])
# sync GPUs and start the timer
accelerator.wait_for_everyone()

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    # store output of generations in dict
    results=dict(
        ko_text=[],
        en_text=[],
        ko_en_outputs=[], 
        en_ko_outputs=[], 
    )

    # have each GPU do inference, prompt by prompt
    for data in tqdm(prompts):
        en_text, ko_text = data
        en_prompt = f"Translate this from English to Korean:\nEnglish: {en_text}\nKorean:"
        ko_prompt = f"Translate this from Korean to English:\nKorean: {ko_text}\nEnglish:"
        
        prompt_tokenized= tokenizer([en_prompt,ko_prompt], return_tensors="pt", padding=True, max_length=1024, truncation=True).to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, num_beams=5, max_new_tokens=1024)

        # remove prompt from output 
        output_tokenized=output_tokenized[:, len(prompt_tokenized["input_ids"][0]):]
        output = tokenizer.batch_decode(output_tokenized, skip_special_tokens=True)

        # store outputs and number of tokens in result{}
        results["en_text"].append(en_text)
        results["ko_text"].append(ko_text)
        results["en_ko_outputs"].append(output[0])
        results["ko_en_outputs"].append(output[1])
    results=[results] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs
gather_results=gather_object(results)
joblib.dump(gather_results, 'result.pkl')