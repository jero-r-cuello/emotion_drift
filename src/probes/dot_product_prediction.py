#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

nested_df = pd.read_pickle("../../data/03_activations/andyzou_situations_microsoft_Phi-3-medium-128k-instruct_20250728_173814.pkl")
train_df = nested_df[nested_df["split"] == "train"].copy()

unique_emotions = train_df["emotion_considered"].unique()
print(f'Found emotions: {unique_emotions}')

## 1. Differences and PCA ##
all_emotion_directions = {}
training_means = {} # This is to normalize the test activations later

# Extracting differences and PCA for each emotion
for emotion in unique_emotions:
    print("-" * 50)
    print(f'Processing the emotion: {emotion.upper()}')
    
    emotion_df = train_df[train_df["emotion_considered"] == emotion].copy()
    
    diffs_for_emotion = []
    prompt_ids_for_emotion = sorted(emotion_df["prompt_id"].unique())
    
    for i in range(0, len(prompt_ids_for_emotion), 2):
        id1 = prompt_ids_for_emotion[i]
        id2 = prompt_ids_for_emotion[i+1]
        
        current_activations = emotion_df[emotion_df["prompt_id"] == id1]["activations"].values[0].last_token_activation
        next_activations = emotion_df[emotion_df["prompt_id"] == id2]["activations"].values[0].last_token_activation
        
        diff = current_activations - next_activations

        # For each layer, normalize the difference vector
        normalized_diffs = {}
        for layer, diff_vector in diff.items():
            norm = np.linalg.norm(diff_vector)
            if norm > 0:
                normalized_vector = diff_vector / norm
            else:
                normalized_vector = diff_vector
            normalized_diffs[layer] = normalized_vector
        
        diffs_for_emotion.append(normalized_diffs)

    # Reorganize the differences, so PCA can be applied layer by layer
    hidden_states_by_layer_list = {}
    for diff_dict in diffs_for_emotion:
        for layer_number, diff_vector in diff_dict.items():
            if layer_number not in hidden_states_by_layer_list:
                hidden_states_by_layer_list[layer_number] = []
            hidden_states_by_layer_list[layer_number].append(diff_vector)
    
    relative_hidden_states = {
        layer: np.array(vectors) 
        for layer, vectors in hidden_states_by_layer_list.items()
    }

    # PCA
    manual_directions_for_emotion = {}
    means_for_emotion = {}

    print("Calculating PCA for each layer...")
    for layer_number, diff_matrix in relative_hidden_states.items():
        pca_model = PCA(n_components=1)
        mean_vector = diff_matrix.mean(axis=0)
        means_for_emotion[layer_number] = mean_vector 

        centered_diffs = diff_matrix - mean_vector
        pca_model.fit(centered_diffs)
        direction_vector = pca_model.components_[0]
        manual_directions_for_emotion[layer_number] = direction_vector
        
    all_emotion_directions[emotion] = manual_directions_for_emotion
    training_means[emotion] = means_for_emotion 
    print(f'{len(manual_directions_for_emotion)} dir. vectors for "{emotion}".')

print("-" * 50)
print("\n Process completed!")
print(f'Total generated vectors: {len(all_emotion_directions) * len(all_emotion_directions[unique_emotions[0]])}')
#%%

## 2. Setting the sign of the vectors for each emotion ## 

# this is to ensure that the vectors point in the right direction for positive examples of the emotion
# We will adjust the sign of the vectors based on the average scores of positive and negative examples
# and then multiply the vectors by the determined sign

final_adjusted_directions = {}

print("Adjusting sign of vectors...")

for emotion, layer_directions in all_emotion_directions.items():
    print("-" * 50)
    print(f'Processing signs for vectors of: {emotion.upper()}')
    
    emotion_df = train_df[train_df["emotion_considered"] == emotion]

    adjusted_directions_for_emotion = {}
    
    for layer_number, v in layer_directions.items():
        scores_pos = []
        scores_neg = []
        
        for index, row in emotion_df.iterrows():
            # Get the activation vector for the current layer
            activation_vector = row["activations"]["last_token_activation"][layer_number]
            
            # Calculate the score as the dot product of the activation vector and the direction vector
            score = np.dot(activation_vector, v)

            # Is it a positive or negative example of the emotion?
            label = row["label"]
            if label == 1:
                scores_pos.append(score)
            else:
                scores_neg.append(score)

        # Determine the sign based on the average scores
        if np.mean(scores_pos) > np.mean(scores_neg):
            sign = 1
        else:
            sign = -1
            
        # Adjust the direction vector by multiplying it by the sign
        v_prime = v * sign
        
        adjusted_directions_for_emotion[layer_number] = v_prime

    final_adjusted_directions[emotion] = adjusted_directions_for_emotion
    
print("-" * 50)
print("\n Sign adjustment completed!")
# %%

## 3. Making inference on the test set ##

print("Initializing the inference process...")

test_df = nested_df[nested_df["split"] == "test"].copy()

test_df["layer_scores"] = [{} for _ in range(len(test_df))] # To store scores for each emotion, layer by layer
test_df["average_scores"] = [{} for _ in range(len(test_df))] # To store average layer scores for each emotion

for index, row in test_df.iterrows():
    for emotion, layer_directions in final_adjusted_directions.items():
        scores_by_layer_for_emotion = {}
        
        # For each layer,
        for layer_number, v_prime in layer_directions.items():
            # a. Extract the activation vector for the current layer
            H_test = row["activations"]["last_token_activation"][layer_number]
            
            # b. Obtain the training mean for this emotion and layer
            train_mean = training_means[emotion][layer_number]
            
            # c. Normalize the test activation vector by subtracting the training mean
            H_test_centered = H_test - train_mean
            
            # d. Calculate dot product to get the score
            score = np.dot(H_test_centered, v_prime)
            
            scores_by_layer_for_emotion[layer_number] = score
            
        test_df.loc[index, "layer_scores"][emotion] = scores_by_layer_for_emotion
        average_score_for_emotion = np.mean(list(scores_by_layer_for_emotion.values()))
        test_df.loc[index, "average_scores"][emotion] = average_score_for_emotion

print("\nInference process completed!")

#%%

## 4. Evaluating the results ##

print("-" * 50)
print("Calculating accuracy...")

correct_predictions_avg = 0
total_predictions = len(test_df)

for index, row in test_df.iterrows():
    emotion_target = row["emotion_considered"]
    emotion_predict_avg = max(row["average_scores"], key=row["average_scores"].get)
    
    if emotion_target == emotion_predict_avg:
        correct_predictions_avg += 1

accuracy_avg = correct_predictions_avg / total_predictions if total_predictions > 0 else 0
print("\n--- General accuracy (all layers mean) ---")
print(f'Acc: {accuracy_avg:.2%} ({correct_predictions_avg}/{total_predictions})')
print("-" * 50)


layer_accuracies = {}

# Take first row to get the format. Take any emotion to get the layer numbers
first_row_scores = test_df.iloc[0]['layer_scores']
any_emotion = list(first_row_scores.keys())[0]
layer_numbers = list(first_row_scores[any_emotion].keys())
        
for layer in layer_numbers:
    correct_predictions_layer = 0
            
    for index, row in test_df.iterrows():
        emotion_target = row['emotion_considered']
                
        scores_for_current_layer = {
            emotion: scores[layer] for emotion, 
            scores in row['layer_scores'].items()}
                
        emotion_predict_layer = max(scores_for_current_layer, key=scores_for_current_layer.get)
                
        if emotion_target == emotion_predict_layer:
            correct_predictions_layer += 1
            
    accuracy_layer = correct_predictions_layer / total_predictions if total_predictions > 0 else 0
    layer_accuracies[layer] = accuracy_layer


print("\n--- Accuracy for each layer ---")
for layer, acc in layer_accuracies.items():
    print(f'Layer {layer:<2}: {acc:.2%}')

best_layer = max(layer_accuracies, key=layer_accuracies.get)
worst_layer = min(layer_accuracies, key=layer_accuracies.get)
    
print("\n--- Overview ---")
print(f'Best layer: {best_layer} (Acc: {layer_accuracies[best_layer]:.2%})')
print(f'Worst layer:  {worst_layer} (Acc: {layer_accuracies[worst_layer]:.2%})')
print("-" * 50)

    
# %%
# Acá hubo una modificación medio rara de Gemini.
# En realidad debería charlar el problema con Jonas y Antonio.

print("-" * 50)
print("Calculando la precisión BINARIA (label=1 vs label=0) para cada emoción...")

# Diccionario para guardar la precisión por emoción
dot_product_accuracies = {}
dot_product_accuracies_by_layer = {emotion: {} for emotion in unique_emotions}

for emotion in unique_emotions:
    
    # Filtramos el test_df para la emoción actual
    emotion_test_df = test_df[test_df['emotion_considered'] == emotion].copy()
    
    # --- a. Evaluación usando el score promediado ---
    correct_predictions_avg = 0
    total_predictions = len(emotion_test_df)
    
    for index, row in emotion_test_df.iterrows():
        # El "score" para esta emoción ya está calculado
        score = row['average_scores'][emotion]
        
        # La predicción es 1 si el score es positivo, 0 si es negativo
        prediction = 1 if score > 0 else 0
        
        # Comparamos con la etiqueta real (label)
        if prediction == row['label']:
            correct_predictions_avg += 1
            
    accuracy_avg = correct_predictions_avg / total_predictions if total_predictions > 0 else 0
    dot_product_accuracies[emotion] = accuracy_avg

    # --- b. Evaluación por capa ---
    for layer in layer_numbers:
        correct_predictions_layer = 0
        for index, row in emotion_test_df.iterrows():
            score = row['layer_scores'][emotion][layer]
            prediction = 1 if score > 0 else 0
            if prediction == row['label']:
                correct_predictions_layer += 1
        
        accuracy_layer = correct_predictions_layer / total_predictions if total_predictions > 0 else 0
        dot_product_accuracies_by_layer[emotion][layer] = accuracy_layer


print("\n--- Precisión BINARIA (Dot Product) por Emoción (usando scores promediados) ---")
for emotion, acc in dot_product_accuracies.items():
    print(f"Emoción '{emotion}': {acc:.2%}")

print("\n--- Mejor Capa por Emoción (Dot Product) ---")
for emotion in unique_emotions:
    best_layer = max(dot_product_accuracies_by_layer[emotion], key=dot_product_accuracies_by_layer[emotion].get)
    best_acc = dot_product_accuracies_by_layer[emotion][best_layer]
    print(f"Emoción '{emotion}': Mejor capa es la {best_layer} con {best_acc:.2%} de precisión.")


print("\n" + "-"*50)
print("Calculando la precisión PROMEDIO POR CAPA (a través de todas las emociones)...")

# Diccionario para guardar el promedio de precisión de cada capa
average_accuracy_per_layer = {}

# Iteramos sobre cada capa
for layer in layer_numbers:
    
    # Recopilamos la precisión de esta capa para cada una de las emociones
    accuracies_for_this_layer = [
        dot_product_accuracies_by_layer[emotion][layer] 
        for emotion in unique_emotions
    ]
    
    # Calculamos el promedio y lo guardamos
    average_accuracy_per_layer[layer] = np.mean(accuracies_for_this_layer)

print("\n--- Precisión Promedio por Capa (a través de todas las emociones) ---")
if average_accuracy_per_layer:
    for layer, acc in average_accuracy_per_layer.items():
        print(f"Capa {layer:<2}: {acc:.2%}")
    
    # Encontrar la mejor capa en general
    best_overall_layer = max(average_accuracy_per_layer, key=average_accuracy_per_layer.get)
    best_overall_acc = average_accuracy_per_layer[best_overall_layer]
    
    print("\n--- Resumen General (Dot Product) ---")
    print(f"Mejor Capa General (promediando emociones): Capa {best_overall_layer} con {best_overall_acc:.2%} de precisión.")
else:
    print("No se pudieron calcular las precisiones promedio por capa.")

print("-" * 50)
# %%
