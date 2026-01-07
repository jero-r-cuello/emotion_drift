# %%
# --- 1. IMPORTACIONES Y CONFIGURACIÓN INICIAL ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Diccionario de mapeo de valencia emocional
sentiment_map_prompt = {
    # --- Emociones Positivas ---
    'love': 'positive', 'joy': 'positive', 'delight': 'positive', 'hope': 'positive',
    'adoration': 'positive', 'affection': 'positive', 'fondness': 'positive', 'liking': 'positive',
    'attraction': 'positive', 'caring': 'positive', 'tenderness': 'positive', 'compassion': 'positive',
    'sentimentality': 'positive', 'desire': 'positive', 'lust': 'positive', 'passion': 'positive',
    'infatuation': 'positive', 'amusement': 'positive', 'bliss': 'positive', 'cheerfulness': 'positive',
    'gaiety': 'positive', 'glee': 'positive', 'jolliness': 'positive', 'joviality': 'positive',
    'enjoyment': 'positive', 'gladness': 'positive', 'happiness': 'positive', 'jubilation': 'positive',
    'elation': 'positive', 'satisfaction': 'positive', 'ecstasy': 'positive', 'euphoria': 'positive',
    'enthusiasm': 'positive', 'zeal': 'positive', 'zest': 'positive', 'excitement': 'positive',
    'thrill': 'positive', 'exhilaration': 'positive', 'contentment': 'positive', 'pleasure': 'positive',
    'pride': 'positive', 'triumph': 'positive', 'eagerness': 'positive', 'optimism': 'positive',
    'enthrallment': 'positive', 'rapture': 'positive', 'relief': 'positive', 'gratitude': 'positive', 
    'approval': 'positive', 'admiration': 'positive', 'trust': 'positive', 'anticipation': 'positive',

    # --- Emociones Negativas ---
    'anger': 'negative', 'sadness': 'negative', 'fear': 'negative', 'agony': 'negative',
    'disgust': 'negative', 'shame': 'negative', 'aggravation': 'negative', 'irritation': 'negative',
    'agitation': 'negative', 'annoyance': 'negative', 'grouchiness': 'negative', 'grumpiness': 'negative',
    'exasperation': 'negative', 'frustration': 'negative', 'rage': 'negative', 'outrage': 'negative',
    'fury': 'negative', 'wrath': 'negative', 'hostility': 'negative', 'ferocity': 'negative',
    'bitterness': 'negative', 'hate': 'negative', 'loathing': 'negative', 'scorn': 'negative',
    'spite': 'negative', 'vengefulness': 'negative', 'dislike': 'negative', 'resentment': 'negative',
    'revulsion': 'negative', 'contempt': 'negative', 'envy': 'negative', 'jealousy': 'negative',
    'torment': 'negative', 'suffering': 'negative', 'hurt': 'negative', 'anguish': 'negative',
    'depression': 'negative', 'despair': 'negative', 'hopelessness': 'negative', 'gloom': 'negative',
    'glumness': 'negative', 'unhappiness': 'negative', 'grief': 'negative', 'sorrow': 'negative',
    'woe': 'negative', 'misery': 'negative', 'melancholy': 'negative', 'dismay': 'negative',
    'disappointment': 'negative', 'displeasure': 'negative', 'guilt': 'negative', 'regret': 'negative',
    'remorse': 'negative', 'alienation': 'negative', 'isolation': 'negative', 'neglect': 'negative',
    'loneliness': 'negative', 'rejection': 'negative', 'homesickness': 'negative', 'defeat': 'negative',
    'dejection': 'negative', 'insecurity': 'negative', 'embarrassment': 'negative', 'humiliation': 'negative',
    'insult': 'negative', 'pity': 'negative', 'sympathy': 'negative', 'alarm': 'negative', 'shock': 'negative', 
    'fright': 'negative', 'horror': 'negative', 'terror': 'negative', 'panic': 'negative', 'hysteria': 'negative', 
    'mortification': 'negative', 'anxiety': 'negative', 'nervousness': 'negative', 'tenseness': 'negative', 
    'uneasiness': 'negative', 'apprehension': 'negative', 'worry': 'negative', 'distress': 'negative', 
    'dread': 'negative', 'disapproval': 'negative',

    # --- Emociones Neutras o Ambiguas ---
    'surprise': 'neutral/ambiguous', 'amazement': 'neutral/ambiguous', 'astonishment': 'neutral/ambiguous',
    'arousal': 'neutral/ambiguous', 'longing': 'neutral/ambiguous', 'neutral': 'neutral/ambiguous',
    'realization': 'neutral/ambiguous', 'curiosity': 'neutral/ambiguous', 'confusion': 'neutral/ambiguous',
    'emotion_target': 'neutral/ambiguous'
}

# --- SIMULACIÓN DE DATOS (REEMPLAZA ESTO CON TU DATAFRAME) ---
# Usaré las claves del diccionario para asegurar que todas las emociones estén presentes
DATA_PATH = "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL.pkl"
nested_df_original = pd.read_pickle(DATA_PATH)
# -----------------------------------------------------------------

columnas_interes = ['ekman_basic_emotions', 'go_emotions', 'plutchik_wheel']

# =============================================================================
# GRÁFICO 1: Barras Horizontales con Etiquetas de Eje Y Coloreadas
# =============================================================================

print("Generando Gráfico 1: Conteos por emoción con etiquetas coloreadas...")

# --- PREPARACIÓN DE DATOS (como en tu script) ---
counts_df = nested_df_original.groupby('emotion_considered')[columnas_interes].count()
total_counts_por_emocion = counts_df.sum(axis=1)
sorted_emotions = total_counts_por_emocion.sort_values(ascending=False).index

df_plot = counts_df.reset_index().melt(
    id_vars='emotion_considered',
    value_vars=columnas_interes,
    var_name='fuente_emocion',
    value_name='cantidad_no_nulos'
)
df_plot['emotion_considered'] = pd.Categorical(
    df_plot['emotion_considered'], 
    categories=sorted_emotions, 
    ordered=True
)

# --- CREACIÓN DEL GRÁFICO ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 30)) # Ajusta la altura si es necesario

ax1 = sns.barplot(
    data=df_plot, 
    x='cantidad_no_nulos', 
    y='emotion_considered', 
    hue='fuente_emocion'
)

# --- MODIFICACIÓN: Colorear etiquetas del eje Y ---
color_map = {'positive': 'green', 'negative': 'red'}
# Iteramos sobre las etiquetas del eje Y (las emociones)
for label in ax1.get_yticklabels():
    emotion = label.get_text()
    # Obtenemos la valencia del diccionario, con un valor por defecto para no causar errores
    valence = sentiment_map_prompt.get(emotion, 'neutral/ambiguous')
    # Obtenemos el color, si no es positivo/negativo, no se asigna color específico (queda negro)
    color = color_map.get(valence)
    if color:
        label.set_color(color)
        label.set_weight('bold') # Opcional: hacerlas negrita para que resalten

# --- PERSONALIZACIÓN Y VISUALIZACIÓN ---
ax1.set_title('Conteo de Emociones por Fuente (con Valencia indicada)', fontsize=22, pad=20)
ax1.set_xlabel('Cantidad de Valores No Nulos', fontsize=14)
ax1.set_ylabel('Emoción Considerada', fontsize=14)
plt.legend(title='Fuente de Emoción', title_fontsize='13', fontsize='11', loc='lower right')
plt.tight_layout()
plt.show()


# =============================================================================
# GRÁFICO 2: Proporciones Apiladas de Nulos/No Nulos por Valencia
# =============================================================================

print("\nGenerando Gráfico 2: Proporciones apiladas por valencia...")

# --- PREPARACIÓN DE DATOS ---
# 1. Añadir la columna de valencia al DataFrame original
df_con_valencia = nested_df_original.copy()
df_con_valencia['valence'] = df_con_valencia['emotion_considered'].map(sentiment_map_prompt)
# Rellenar cualquier posible valencia faltante
df_con_valencia['valence'].fillna('neutral/ambiguous', inplace=True)

# 2. Transformar el DataFrame a formato largo para facilitar el conteo y graficado
df_melted = df_con_valencia.melt(
    id_vars=['valence'], 
    value_vars=columnas_interes,
    var_name='fuente_emocion',
    value_name='valor'
)

# 3. Crear una columna que indique si el valor es Nulo o No Nulo
df_melted['Tipo'] = np.where(df_melted['valor'].isnull(), 'Nulo', 'No Nulo')


# --- CREACIÓN DEL GRÁFICO ---
# Usaremos histplot de seaborn, que es perfecto para proporciones apiladas
sns.set_theme(style="ticks")
g = sns.displot(
    data=df_melted,
    x='valence',          # Las valencias en el eje X
    hue='Tipo',           # El color de la barra indica si es Nulo o No Nulo
    col='fuente_emocion', # Crea un subgráfico para cada fuente de emoción
    multiple='fill',      # ¡La clave! Apila las barras y normaliza a 1 (100%)
    shrink=0.8,           # Reduce el ancho de las barras para separarlas
    palette={'No Nulo': '#4CAF50', 'Nulo': '#F44336'} # Colores verde/rojo para consistencia
)

# --- PERSONALIZACIÓN Y VISUALIZACIÓN ---
g.fig.suptitle('Proporción de Valores Nulos vs. No Nulos por Valencia y Fuente', y=1.03, fontsize=16)
g.set_axis_labels("Valencia Emocional", "Proporción")
g.set_titles("Fuente: {col_name}")
g.legend.set_title("Tipo de Valor")
plt.tight_layout()
plt.show()
# %%
