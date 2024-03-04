import datasets
from tqdm import tqdm

eng_dataset = datasets.load_dataset('gsarti/flores_101', 'eng')
kor_dataset = datasets.load_dataset('gsarti/flores_101', 'kor')
kor_dataset = kor_dataset['dev']['sentence'] + kor_dataset['devtest']['sentence']
eng_dataset = eng_dataset['dev']['sentence'] + eng_dataset['devtest']['sentence']

from openai import OpenAI
client = OpenAI('your_key') # maybe you will need 20$

checkpoint_save_response = {}
checkpoint_save_response['ko-en'] = []
checkpoint_save_response['en-ko'] = []
checkpoint_save_response['ko'] = []
checkpoint_save_response['en'] = []

for ko, en in tqdm(zip(kor_dataset,eng_dataset), total=len(kor_dataset)):
    sourcelanguage = 'Korean'
    targetlanguage = 'English'
    sourcesentence = ko
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {
          "role": "system",
          "content": "You areahelpful translator and only output theresult."
        },
        {
          "role": "user",
          "content": f"### Translatethis from {sourcelanguage} to {targetlanguage}, {sourcelanguage}:\n{sourcesentence}\n### {targetlanguage}:"
        }
      ],
      temperature=0.5,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    checkpoint_save_response['ko'].append(ko)
    checkpoint_save_response['ko-en'].append(response)

    sourcelanguage = 'English'
    targetlanguage = 'Korean'
    sourcesentence = en
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {
          "role": "system",
          "content": "You areahelpful translator and only output theresult."
        },
        {
          "role": "user",
          "content": f"### Translatethis from {sourcelanguage} to {targetlanguage}, {sourcelanguage}:\n{sourcesentence}\n### {targetlanguage}:"
        }
      ],
      temperature=0.5,
      max_tokens=1024,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    checkpoint_save_response['en-ko'].append(response)
    checkpoint_save_response['en'].append(en)

import joblib
joblib.dump(checkpoint_save_response, 'gpt4.pkl')
