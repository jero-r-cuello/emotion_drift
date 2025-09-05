#%%
import os
from transformers import pipeline, AutoConfig
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd

# Define los nombres de los modelos
ekman_annotator = "monologg/bert-base-cased-goemotions-ekman"
go_emotions_annotator = "monologg/bert-base-cased-goemotions-original"

ekman_classifier = pipeline("text-classification", 
                      model=ekman_annotator, 
                      top_k=None,
                      device=-1) # Usar CPU

go_emotions_classifier = pipeline("text-classification", 
                      model=go_emotions_annotator, 
                      top_k=None,
                      device=-1) # Usar CPU

generated_responses = pd.read_csv("/home/jcuello/emotion_drift/data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv")["response_text"].tolist()

results_data = []
print("Processing responses...")
for response in generated_responses:
     all_ekman_results = ekman_classifier(response)[0]
     all_go_emotions_results = go_emotions_classifier(response)[0]
     
     # 2. Filtrar las etiquetas de Ekman con score > 0.2
     filtered_ekman_results = [result for result in all_ekman_results if result['score'] >= 0.2]
     
     # 3. Aplicar la condición: si la lista filtrada no está vacía, usarla.
     #    Si está vacía, quedarse solo con el primer elemento (el de mayor score).
     if len(filtered_ekman_results) > 0:
         final_ekman_labels = [res['label'] for res in filtered_ekman_results]
         final_ekman_scores = [res['score'] for res in filtered_ekman_results]
     else:
         final_ekman_labels = [all_ekman_results[0]['label']]
         final_ekman_scores = [all_ekman_results[0]['score']]

     # 4. Repetir el mismo proceso para GoEmotions
     filtered_go_emotions_results = [result for result in all_go_emotions_results if result['score'] >= 0.2]
     
     if len(filtered_go_emotions_results) > 0:
         final_go_emotions_labels = [res['label'] for res in filtered_go_emotions_results]
         final_go_emotions_scores = [res['score'] for res in filtered_go_emotions_results]
     else:
         final_go_emotions_labels = [all_go_emotions_results[0]['label']]
         final_go_emotions_scores = [all_go_emotions_results[0]['score']]

     # --- Termina la lógica modificada ---

     # 5. Añadir los resultados (que ahora pueden ser listas) al conjunto de datos
     results_data.append({
          "response_text": response,
          "model": "monologg/bert-base-cased",
          "ekman_labels": final_ekman_labels,
          "ekman_justification": final_ekman_scores, # Ahora contiene una lista de scores
          "go_emotions_labels": final_go_emotions_labels,
          "go_emotions_justification": final_go_emotions_scores # Ahora contiene una lista de scores
        })

print("Responses processed. Saving csv...")
df = pd.DataFrame(results_data)
print(df)
df.to_csv("/home/jcuello/emotion_drift/data/04_annotated/bert_annotators_1.csv")
print("Saved.")
# %%
