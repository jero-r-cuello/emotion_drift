# %%
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import re
import numpy as np
import statsmodels.formula.api as smf
from word2number import w2n
import matplotlib.pyplot as plt
import seaborn as sns

# ### Emotional assessment

def extraer_rating(texto):
    """
    Extrae un rating numérico (1-9) de una cadena de texto.
    
    Maneja:
    - Números como dígitos y palabras.
    - Ignora el contexto de la escala "1 a 9".
    - Decide qué hacer con múltiples números.
    """
    # 1. Manejar casos vacíos o no-texto
    if not isinstance(texto, str):
        return None

    # 2. Pre-procesamiento: Limpiar el ruido común
    # Convertir a minúsculas para unificar
    texto_procesado = texto.lower()
    
    # Substract common phrases including numbers that produce noise
    # "1 to 9" or "1-9" and their derivations
    texto_procesado = re.sub(r'1\s*to\s*9|1\s*-\s*9|1\s*and\s*9', '', texto_procesado)
    texto_procesado = re.sub(r'1\s*-\s|2\s*-\s|3\s*-\s|4\s*-\s|5\s*-\s|6\s*-\s|7\s*-\s|8\s*-\s|9\s*-\s', '', texto_procesado)

    # "out of 9" (as in 'I would say 3 out of 9')
    texto_procesado = re.sub(r'out\s*of\s*9', '', texto_procesado)
    texto_procesado = re.sub(r'/\s*9', '', texto_procesado)
    # The (?:the\s+)? part makes "the " optional, matching both versions of the phrase.
    texto_procesado = re.sub(r'1\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?absence\s*|1\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?minimum\s*', '', texto_procesado)
    texto_procesado = re.sub(r'9\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?maximum\s*', '', texto_procesado)
    texto_procesado = re.sub(r'1\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?most\s*negative\s*', '', texto_procesado)
    texto_procesado = re.sub(r'9\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?maximum\s*positive\s*', '', texto_procesado)
    texto_procesado = re.sub(r'1\s*represents\s*', '', texto_procesado)
    texto_procesado = re.sub(r'9\s*represents\s*', '', texto_procesado)
    texto_procesado = re.sub(r'1\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?negative\s*', '', texto_procesado)
    texto_procesado = re.sub(r'9\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?positive\s*', '', texto_procesado)
    texto_procesado = re.sub(r'1\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?calm\s*', '', texto_procesado)
    texto_procesado = re.sub(r'9\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?excited\s*', '', texto_procesado)
    texto_procesado = re.sub(r'3\s*word\s*|3\s*words\s*', '', texto_procesado)

    # Specific of some prompts
    texto_procesado = re.sub(r'6\s*hours\s*', '', texto_procesado)
    texto_procesado = re.sub(r'rather\s*than\s*(?:a\s*)?9', '', texto_procesado)

    # 3. Convertir números escritos en palabras a dígitos
    # Esto es un bloque try-except porque w2n puede fallar si no hay números en palabras.
    try:
        # w2n.word_to_num convierte "un cinco" a "un 5"
        texto_procesado = w2n.word_to_num(texto_procesado)
    except ValueError:
        # Si no hay palabras numéricas, no hacemos nada.
        pass

    # 4. Encontrar TODOS los números restantes que estén en el rango de 1 a 9
    # La expresión regular \b[1-9]\b busca un único dígito del 1 al 9 que sea
    # una "palabra completa" (\b es un límite de palabra).
    # Esto evita que '19' se interprete como '1' y '9'.
    candidatos = re.findall(r'\b[1-9]\b', str(texto_procesado))
    
    # 5. Lógica de decisión para seleccionar el número correcto
    if not candidatos:
        # No se encontró ningún número válido.
        return 0 # equals None
    else:
        candidatos_int = [int(num) for num in candidatos]
        ratings_unicos = list(dict.fromkeys(candidatos_int))

        if len(ratings_unicos) == 1:
            return ratings_unicos[0]
        elif len(ratings_unicos) == 9:
            return 0
        else:
            if ratings_unicos == [1, 2, 3, 4, 5] or ratings_unicos == [1, 2, 3]: # Probably a list of smt
                return 0
            else:
                return ratings_unicos[0] # In the notebook this indexing is not present

# %%

assessment_list = ["fear_intensity","sadness_intensity",
                   "joy_intensity","disgust_intensity",
                   "anger_intensity","surprise_intensity",
                   "SAM_valence","SAM_arousal"]
final_assessment_df = None

for assessment_used in assessment_list:
    assessment_path = f"/home/jcuello/emotion_drift/data/02_generated/outputs_{assessment_used}_assessment_Llama-2-7b-chat-hf.jsonl"

    assessment_df = pd.read_json(assessment_path, lines=True)

    if final_assessment_df is None:
        final_assessment_df = assessment_df[["prompt_key"]].copy()

    assessment_df[f'{assessment_used}_ratings'] = assessment_df["generated_text"].apply(extraer_rating)

    rating_text_pairs = assessment_df.loc[:,[f'{assessment_used}_ratings',"generated_text"]]

    for idx, pair in rating_text_pairs.iterrows():
        print("="*30)
        print(f'\nExtracted {pair[f"{assessment_used}_ratings"]} for text: \n{pair["generated_text"]}')


    print(assessment_df[f'{assessment_used}_ratings'].value_counts())


    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))

    ax = sns.countplot(x=f'{assessment_used}_ratings', data=assessment_df, palette="tab10")

    ax.set_title(f'Frequency of {assessment_used} ratings', fontsize=16)
    ax.set_xlabel('Extracted rating', fontsize=12)
    ax.set_ylabel('Freq (Count)', fontsize=12)
    plt.tight_layout() 

    plt.show()
    plt.close()

    # Plot without None ratings
    valid_ratings = assessment_df[assessment_df[f'{assessment_used}_ratings']!=0]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))

    ax = sns.countplot(x=f'{assessment_used}_ratings', data=valid_ratings, palette="tab10")

    ax.set_title(f'Frequency of {assessment_used} valid ratings', fontsize=16)
    ax.set_xlabel('Extracted rating', fontsize=12)
    ax.set_ylabel('Freq (Count)', fontsize=12)
    plt.tight_layout() 

    plt.show()
    plt.close()

    final_assessment_df = pd.merge(left=final_assessment_df, right=assessment_df[["prompt_key",f'{assessment_used}_ratings']], on="prompt_key", how="left")

columnas_de_emociones = [
    'fear_intensity_ratings', 'sadness_intensity_ratings', 
    'joy_intensity_ratings', 'disgust_intensity_ratings',
    'anger_intensity_ratings', 'surprise_intensity_ratings',
    'SAM_valence_ratings', 'SAM_arousal_ratings']

# Long df to create a plot
df_long = pd.melt(
    final_assessment_df, 
    id_vars=['prompt_key'],  # Columna(s) a mantener como identificadores
    value_vars=columnas_de_emociones,  # Columnas a transformar en filas
    var_name='emotion',  # Nombre de la nueva columna para las emociones
    value_name='rating'  # Nombre de la nueva columna para las puntuaciones
)

# Filter it to exclude None ratings
df_filtered = df_long[df_long['rating'] > 0].copy()
n_counts = df_filtered['emotion'].value_counts()

# Bar plot
plt.figure(figsize=(12, 8))

ax = sns.barplot(
    data=df_filtered, 
    x='emotion', 
    y='rating', 
    palette='tab10',
)

for i, patch in enumerate(ax.patches):
    emotion_name = n_counts.index[i]
    # Obtenemos el conteo 'n' para esa emoción
    count = n_counts[emotion_name]
    
    # Creamos el texto de la etiqueta
    label = f"n={count}"
    
    # Calculamos la posición x e y para el texto
    x = patch.get_x() + patch.get_width() / 2  # Centrado en la barra
    y = patch.get_height() + 0.1               # Un poco por encima de la barra
    
    # Añadimos el texto al gráfico
    ax.text(x, y, label, ha='center', va='bottom', color='black', fontsize=11, weight='bold')

ax.set_title('Mean rating per emotion (Only valid ratings)', fontsize=16)
ax.set_xlabel('Emotion / Dimension', fontsize=12)
ax.set_ylabel('Mean rating', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 10) 
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))

# Violin plot
ax = sns.violinplot(
    data=df_filtered, 
    x='emotion', 
    y='rating', 
    palette='tab10',
    inner='quartile',
    cut=0
)

for violin in ax.collections:
    violin.set_alpha(0.6) 

sns.stripplot(
    data=df_filtered,
    x='emotion',
    y='rating',
    color='white',         # Puntos blancos que resaltan
    edgecolor='gray',      # Con un borde gris
    linewidth=0.5,
    jitter=0.15,           # Un poco de "ruido" horizontal para que no se superpongan
    size=4,                # Tamaño de los puntos
    ax=ax
)

ax.set_ylim(0, 11.5) 

# Itera sobre cada categoría en el eje x
for i, emotion_name in enumerate(n_counts.index):
    # Obtenemos el conteo 'n' para esa emoción
    count = n_counts[emotion_name]
    label = f"n={count}"
    
    # Añadimos el texto en la posición x=i, y en una altura fija (ej. 10.5)
    ax.text(i, 10.5, label, ha='center', va='bottom', color='black', fontsize=12, weight='bold')

ax.set_title('Distribución de Puntuaciones por Emoción (Ratings > 0)', fontsize=18)
ax.set_xlabel('Emoción', fontsize=14)
ax.set_ylabel('Distribución de Puntuaciones', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Correlation matrix of emotions
final_assessment_df.replace(0,np.nan,inplace=True)
emotion_columns = final_assessment_df.columns[1:]
correlation_matrix = final_assessment_df[emotion_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=.5
)
plt.title('Matriz de Correlación entre Emociones', fontsize=18)
plt.show()

# Scatter plot of circumplex model of affect
df_only_valence = final_assessment_df[final_assessment_df["SAM_valence_ratings"].notna() & final_assessment_df["SAM_arousal_ratings"].isna()]
df_only_arousal = final_assessment_df[final_assessment_df["SAM_valence_ratings"].isna() & final_assessment_df["SAM_arousal_ratings"].notna()]


# --- 3. CREAR EL GRÁFICO AVANZADO ---
plt.figure(figsize=(12, 12))

# --- CAPA DE FONDO: Datos Parciales ---
# Usamos un scatter plot normal, pero con jitter en una sola dirección
# Ploteamos los que solo tienen Valencia en el eje X
plt.scatter(
    df_only_valence["SAM_valence_ratings"] + np.random.uniform(-0.1, 0.1, size=len(df_only_valence)), # Jitter en X
    np.full(len(df_only_valence), -0.2), # Posición fija en Y (fuera del área principal)
    color='lightgray', s=50, alpha=0.8, label='Solo Valencia'
)
# Ploteamos los que solo tienen Arousal en el eje Y
plt.scatter(
    np.full(len(df_only_arousal), -0.2), # Posición fija en X
    df_only_arousal["SAM_arousal_ratings"] + np.random.uniform(-0.1, 0.1, size=len(df_only_arousal)), # Jitter en Y
    color='lightgray', s=50, alpha=0.8, label='Solo Arousal'
)


# --- CAPA PRINCIPAL: Datos Completos ---
# Usaremos regplot de Seaborn que maneja jitter y alpha de forma nativa
# Lo usamos sin la línea de regresión (fit_reg=False)
sns.regplot(
    data=final_assessment_df[["SAM_valence_ratings","SAM_arousal_ratings"]].dropna(),
    x="SAM_valence_ratings",
    y="SAM_arousal_ratings",
    fit_reg=False, # No queremos la línea de regresión
    x_jitter=0.2,  # Cantidad de jitter horizontal
    y_jitter=0.2,  # Cantidad de jitter vertical
    scatter_kws={
        'alpha': 0.4, # Transparencia de los puntos
        's': 100,     # Tamaño de los puntos
        'color': 'dodgerblue',
        'edgecolor': 'white',
    },
    label='Datos Completos'
)

# --- 4. FORMATEO (líneas, etiquetas de cuadrantes, etc.) ---
mid_point = 4.5
plt.axhline(mid_point, color='grey', linestyle='--')
plt.axvline(mid_point, color='grey', linestyle='--')

plt.text(mid_point + 0.2, mid_point + 0.2, 'Alta Energía, Positivo', fontsize=12, color='darkgreen')
plt.text(mid_point - 0.2, mid_point + 0.2, 'Alta Energía, Negativo', fontsize=12, color='darkred', ha='right')
plt.text(mid_point - 0.2, mid_point - 0.2, 'Baja Energía, Negativo', fontsize=12, color='darkblue', ha='right', va='top')
plt.text(mid_point + 0.2, mid_point - 0.2, 'Baja Energía, Positivo', fontsize=12, color='darkorange', va='top')

plt.title('Mapa Afectivo Bidimensional (Mostrando Overplotting y Datos Parciales)', fontsize=18)
plt.xlabel('Valencia (Negativo a Positivo)', fontsize=12)
plt.ylabel('Arousal (Calma a Excitación)', fontsize=12)
plt.xlim(-0.5, 9.5)
plt.ylim(-0.5, 9.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend() # Muestra la leyenda para identificar los tipos de puntos
plt.show()
# %%
