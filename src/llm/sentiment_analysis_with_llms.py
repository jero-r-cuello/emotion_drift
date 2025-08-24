# sentiment_analysis_with_llms.py
#%%
import google.generativeai as genai
import os
import time
import pandas as pd

API_KEY = "AIzaSyBUDAMOCvN1vdKkNMWTYf5cNqaXo5Ysgfo" #!! OJO NO LA DEJES ACÁ
genai.configure(api_key=API_KEY)

df_out_of_domain = pd.read_csv("/home/jcuello/emotion_drift/data/01_stimuli/testing_out_of_domain/out_of_domain.csv")
situations = df_out_of_domain["situation"]

label_space = ["happiness","fear","sadness","anger","surprise","disgust"]
prompts = [f'You are an emotion classification assistant. Your task is to predict the intensity of emotions expressed in the input sentences for the following categories: {label_space}. \nThe intensity levels range from:\n  -1: Abscense of emotion\n  -9: Maximum intensity of emotion\nProvide the intensity for each emotion as a number between 1 and 9\nAlways output the result in the format "[emotion 1: score 1, emotion 2: score 2, ..., emotion n: score n]", where score is the predicted intensity for each emotion. \nEnsure your prediction reflects the perceived intensity of the input sentence for all emotions, even if some intensities are 1.\nInput sentence: {situation}' for situation in situations]

model = genai.GenerativeModel('gemini-2.5-flash')
gen_config = {
    "temperature": 0.0, #!! Set to 0 as in previous work that states temperature value
    "top_p": 0.95, #!! What should be the value of this??
    "top_k": 40, #!! And this??
}

petitions_per_minute = 9
min_interval = 60.0 / petitions_per_minute

responses = []
for idx, prompt in enumerate(prompts):
    start_time = time.monotonic()
    
    try:
        response = model.generate_content(prompt,
                                          generation_config=gen_config).text
        print(f"\n===GENERATED ANNOTATION {idx}===\n {response}")
        
    except Exception as e:
        print(f"An error ocurred: {e}")
        time.sleep(min_interval)
        continue

    end_time = time.monotonic()
    tiempo_transcurrido = end_time - start_time
    
    tiempo_de_espera = min_interval - tiempo_transcurrido
    
    if tiempo_de_espera > 0:
        time.sleep(tiempo_de_espera)

    responses.append(response)

# %%
