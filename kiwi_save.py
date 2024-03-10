import joblib
from comet import download_model, load_from_checkpoint

output  = joblib.load('./output.pkl')

datas = []

for data in output:
    datas.append(
        {
            "src": data['en'], 
            "mt": data['ko'], 
        }
    )
    datas.append(
        {
            "src": data['ko'], 
            "mt": data['en'], 
        }
    )

    
    datas.append(
        {
            "src": data['en'], 
            "mt": data['alma_ko'], 
        }
    )
    datas.append(
        {
            "src": data['ko'], 
            "mt": data['alma_en'], 
        }
    )

    
    datas.append(
        {
            "src": data['en'], 
            "mt": data['gpt4_ko'], 
        }
    )
    datas.append(
        {
            "src": data['ko'], 
            "mt": data['gpt4_en'], 
        }
    )

model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
model = load_from_checkpoint(model_path)
model_output = model.predict(datas, batch_size=1, gpus=1)
joblib.dump(model_output, './kiwi.pkl')