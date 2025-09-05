# generated_prompt_variance_checks.py
#%% Function definition. To run the script go below
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA # Importamos PCA
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import os


def preprocessing(file_name, n_prompts_expected, save):
    if file_name.endswith(".txt"):
        with open(file_name, 'r', encoding='utf-8') as f:
            texto_completo = f.read()

        emotion_chunks = texto_completo.split("===PROMPT EMOTION===")

        prompts_per_emotion_dict = {}
        for chunk in emotion_chunks[1:]:
            splited_chunk = chunk.split("===GENERATED PROMPTS===")

            emotion = splited_chunk[0]
            clean_emotion = emotion.split('\n')[1].strip()

            prompts = splited_chunk[1]
            lines = prompts.split('\n')

            prompt_list = []
            for line in lines:
                # Quitamos espacios en blanco al principio y al final
                clean_line = line.strip()

                # Comprobamos que la línea no esté vacía y que empiece con un número
                if clean_line and clean_line[0].isdigit():
                    # Dividimos la línea por el primer ". " para separar el número del texto
                    # El ", 1" asegura que solo divida en la primera ocurrencia
                    prompt = clean_line.split('. ', 1)[1]
                    prompt_list.append(prompt.strip())

            prompts_per_emotion_dict[clean_emotion] = prompt_list

        filtered_prompts_dict = {}
        excluded_emotions = []

        print("--- Verificando la longitud de las listas de prompts ---")
        for emotion, prompts in prompts_per_emotion_dict.items():
            current_len = len(prompts)
            print(f"Emoción: {emotion}, Longitud: {current_len}")
            if current_len == n_prompts_expected:
                filtered_prompts_dict[emotion] = prompts
            else:
                # Guardamos la información de las emociones excluidas para reportarlas
                excluded_emotions.append((emotion, current_len))

        if excluded_emotions:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("AVISO: Se excluyeron las siguientes emociones por no tener la longitud esperada de prompts:")
            for emotion, length in excluded_emotions:
                print(f"- '{emotion}': Se encontraron {length} prompts en lugar de los {n_prompts_expected} esperados.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

        if save:
            try:
                    # Usamos el diccionario filtrado, no el original
                df_prompts = pd.DataFrame(filtered_prompts_dict)
                output_path = "/home/jcuello/emotion_drift/data/01_stimuli/llm_focused_situations/generated_prompts.csv"
                df_prompts.to_csv(output_path, index=False) # Usar index=False es buena práctica aquí
                print(f"DataFrame guardado correctamente en {output_path}")
            except Exception as e:
                # Este except ahora capturará otros errores inesperados
                print(f"Ocurrió un error inesperado al crear o guardar el DataFrame: {e}")

        all_prompts = [
            prompt 
            for prompts_list in filtered_prompts_dict.values() 
            for prompt in prompts_list
        ]
        
        return all_prompts
    
    elif file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
        prompts_per_emotion_dict = df.groupby('emotion_target')['generated_prompt'].apply(list).to_dict()

        filtered_prompts_dict = {}
        excluded_emotions = []

        for emotion, prompts in prompts_per_emotion_dict.items():
            filtered_prompts_dict[emotion] = [
                prompt for prompt in prompts if not prompt.startswith("JSON Decode Error")
                ]
            
        if save:
            try:
                # Creamos un DataFrame a partir del diccionario filtrado
                # Las emociones serán las columnas y los prompts las filas
                df_prompts = pd.DataFrame(filtered_prompts_dict)
                output_path = "/home/jcuello/emotion_drift/data/01_stimuli/llm_focused_situations/generated_prompts.csv"
                df_prompts.to_csv(output_path, index=False)
                print(f"DataFrame guardado correctamente en {output_path}")
            except Exception as e:
                print(f"Ocurrió un error inesperado al crear o guardar el DataFrame: {e}")

        # Creamos una lista única con todos los prompts de las emociones que sí cumplieron el criterio
        all_prompts = [
            prompt 
            for prompts_list in filtered_prompts_dict.values() 
            for prompt in prompts_list
        ]

        print("Total prompts after filtering: ", len(all_prompts))        
        return all_prompts

def analisis_sintactico(lista_oraciones, n_of_generation, save):
    print("--- Starting Syntactic Analysis ---")

    

    # 1. Longitud de las oraciones
    longitudes = [len(oracion.split()) for oracion in lista_oraciones]
    df_longitudes = pd.DataFrame(longitudes, columns=['longitud'])
    
    print("\nLenght of sentences stats:")
    print(df_longitudes.describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_longitudes, x='longitud', kde=True, bins=25)
    plt.title(f'Distribution of lenght of sentences (generated prompts per emotion by each model = {n_of_generation})')
    plt.xlabel('Count of words')
    plt.ylabel('Frequency')
    plt.tight_layout()
    if save:
        os.makedirs("/home/jcuello/emotion_drift/figures/generated_prompt_analysis", exist_ok=True)
        plt.savefig(f"/home/jcuello/emotion_drift/figures/generated_prompt_analysis/sentence_lengths_{n_of_generation}.png",
                    dpi=300)
    plt.show()

    # 2. Palabras de inicio
    palabras_inicio = [oracion.split()[0].lower().strip(".,¡¿") for oracion in lista_oraciones if oracion]
    conteo_inicio = Counter(palabras_inicio).most_common(25)
    
    print(f"\n{len(conteo_inicio)} most common starter words:")
    print(conteo_inicio)

    df_inicio = pd.DataFrame(conteo_inicio, columns=['palabra', 'frecuencia'])
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_inicio, x='frecuencia', y='palabra', palette='tab10')
    plt.title(f'25 most common starter words (generated prompts per emotion by each model = {n_of_generation})')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    if save:
        os.makedirs("/home/jcuello/emotion_drift/figures/generated_prompt_analysis", exist_ok=True)
        plt.savefig(f"/home/jcuello/emotion_drift/figures/generated_prompt_analysis/top_starter_words_{n_of_generation}.png",
                    dpi=300)
    plt.show()

def word_frequency_analysis(sentence_list, n_of_generation, save):
    """
    Analyzes the frequency of words (unigrams) and word pairs (bigrams)
    ignoring English stop words.
    """
    print("\n--- Starting Word Frequency Analysis ---")
    
    

    # Get the list of English stop words
    stop_words = set(stopwords.words('english'))
    
    all_words = []
    # Preprocess all sentences
    for sentence in sentence_list:
        # Convert to lowercase and remove punctuation
        clean_text = re.sub(r'[^\w\s]', '', sentence.lower())
        # Tokenize (split into words) and filter stop words
        words = [word for word in clean_text.split() if word not in stop_words]
        all_words.extend(words)
        
    # 1. Unigram count (individual words)
    unigram_counts = Counter(all_words).most_common(25)
    print("\nTop 25 most common words (non-stopwords):")
    print(unigram_counts)
    
    df_unigrams = pd.DataFrame(unigram_counts, columns=['word', 'frequency'])
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_unigrams, x='frequency', y='word', palette='plasma')
    plt.title(f'Top 25 Most Common Words (excluding Stop Words, generated prompts per emotion by each model = {n_of_generation})')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    if save:
        os.makedirs("/home/jcuello/emotion_drift/figures/generated_prompt_analysis", exist_ok=True)
        plt.savefig(f"/home/jcuello/emotion_drift/figures/generated_prompt_analysis/top_words_{n_of_generation}.png",
                    dpi=300)
    plt.show()

    # 2. Bigram count (word pairs)
    # Generate bigrams from the already filtered word list
    bigram_list = list(ngrams(all_words, 2))
    bigram_counts = Counter(bigram_list).most_common(15)
    
    # Format bigrams to be more readable for the plot
    formatted_bigrams = [(' '.join(bigram), freq) for bigram, freq in bigram_counts]
    
    print("\nTop 15 most common bigrams:")
    print(formatted_bigrams)

    df_bigrams = pd.DataFrame(formatted_bigrams, columns=['bigram', 'frequency'])
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_bigrams, x='frequency', y='bigram', palette='magma')
    plt.title(f'Top 15 Most Common Bigrams (generated prompts per emotion by each model = {n_of_generation})')
    plt.xlabel('Frequency')
    plt.ylabel('Bigram')
    plt.tight_layout()
    if save:
        os.makedirs("/home/jcuello/emotion_drift/figures/generated_prompt_analysis", exist_ok=True)
        plt.savefig(f"/home/jcuello/emotion_drift/figures/generated_prompt_analysis/top_bigrams_{n_of_generation}.png",
                    dpi=300)
    plt.show()

        # trigram count
    # Generate trigrams from the already filtered word list
    trigram_list = list(ngrams(all_words, 3))
    trigram_counts = Counter(trigram_list).most_common(15)
    
    # Format bigrams to be more readable for the plot
    formatted_trigrams = [(' '.join(trigram), freq) for trigram, freq in trigram_counts]
    
    print("\nTop 15 most common trigrams:")
    print(formatted_trigrams)

    df_trigrams = pd.DataFrame(formatted_trigrams, columns=['trigram', 'frequency'])
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_trigrams, x='frequency', y='trigram', palette='magma')
    plt.title(f'Top 15 Most Common trigrams (generated prompts per emotion by each model = {n_of_generation})')
    plt.xlabel('Frequency')
    plt.ylabel('trigram')
    plt.tight_layout()
    if save:
        os.makedirs("/home/jcuello/emotion_drift/figures/generated_prompt_analysis", exist_ok=True)
        plt.savefig(f"/home/jcuello/emotion_drift/figures/generated_prompt_analysis/top_trigrams_{n_of_generation}.png",
                    dpi=300)
    plt.show()

        # 4-gram count 
    # Generate 4-grams from the already filtered word list
    four_gram_list = list(ngrams(all_words, 4))
    four_gram_counts = Counter(four_gram_list).most_common(15)
    
    # Format four_grams to be more readable for the plot
    formatted_four_grams = [(' '.join(four_gram), freq) for four_gram, freq in four_gram_counts]
    
    print("\nTop 15 most common 4-grams:")
    print(formatted_four_grams)

    df_four_grams = pd.DataFrame(formatted_four_grams, columns=['four_gram', 'frequency'])
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_four_grams, x='frequency', y='four_gram', palette='magma')
    plt.title(f'Top 15 Most Common 4-grams (generated prompts per emotion by each model = {n_of_generation})')
    plt.xlabel('Frequency')
    plt.ylabel('4-gram')
    plt.tight_layout()
    if save:
        os.makedirs("/home/jcuello/emotion_drift/figures/generated_prompt_analysis", exist_ok=True)
        plt.savefig(f"/home/jcuello/emotion_drift/figures/generated_prompt_analysis/four_gram_{n_of_generation}.png",
                    dpi=300)
    plt.show()

# --- ANÁLISIS SEMÁNTICO ---
def semantic_analysis(lista_oraciones, n_of_generation, save):
    print("\n--- Starting semantic analysis ---")

    

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Generating embeddings...")
    embeddings = model.encode(lista_oraciones, show_progress_bar=True)
    print(f"{embeddings.shape[0]} embeddings of {embeddings.shape[1]} dimension generated.")
    
    print("\nRunning PCA...")
    pca = PCA(n_components=2, random_state=42)
    resultados_2d = pca.fit_transform(embeddings)
    
    print(f"Explained variance by the two components: {pca.explained_variance_ratio_.sum():.2%}")

    df_plot = pd.DataFrame(resultados_2d, columns=['PC1', 'PC2'])
    df_plot['oracion'] = lista_oraciones
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="PC1", y="PC2", data=df_plot, alpha=0.6, s=50)
    plt.title(f'Embeddings (PCA, generated prompts per emotion by each model = {n_of_generation})')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.tight_layout()
    if save:
        os.makedirs("/home/jcuello/emotion_drift/figures/generated_prompt_analysis", exist_ok=True)
        plt.savefig(f"/home/jcuello/emotion_drift/figures/generated_prompt_analysis/embeddings_pca_{n_of_generation}.png",
                    dpi=300)
    plt.show()

def full_pipeline(file_name, n_of_generation, save=True):
    all_prompts = preprocessing(file_name, n_of_generation, save)
    analisis_sintactico(all_prompts, n_of_generation,save)
    word_frequency_analysis(all_prompts, n_of_generation,save)
    semantic_analysis(all_prompts, n_of_generation,save)

#%%    
nombre_archivo = "/home/jcuello/emotion_drift/data/01_stimuli/generated_prompts/generated_emotional_prompts_batched.csv"

full_pipeline(nombre_archivo, 50, True)
print("\n--- Análisis Completo ---")

# %%
