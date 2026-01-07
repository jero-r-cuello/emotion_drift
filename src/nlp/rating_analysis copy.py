# %%
import pandas as pd
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from word2number import w2n
from wordcloud import WordCloud
import statsmodels.formula.api as smf
from patsy import bs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap 
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer # Necesario para BERTopic customizado
from scipy import stats  
import scikit_posthocs as sp
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet

# Descargar recursos de NLTK si no existen
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("[INFO] Descargando recursos de NLTK...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt_tab') # A veces necesario en versiones nuevas
    nltk.download('averaged_perceptron_tagger_eng')

# ==========================================
# 1. CONFIGURACIÓN Y CONSTANTES
# ==========================================

# --- NUEVO: DEFINICIÓN ROBUSTA DE STOPWORDS ---
# Palabras que aparecen por el prompt o la naturaleza del experimento y no aportan semántica específica
EXPERIMENTAL_STOPWORDS = set(stopwords.words('english'))
custom_stops = {
    # Términos del prompt/setup
    'feel', 'feeling', 'felt', 'emotion', 'emotional', 'emotions',
    'im', 'ive', 'would', 'could', 'response', 'category', 'answer', 
    'word', 'state', 'sense', 'describe', 'description', 'sentence',
    'person', 'people', 'situation', 'scenario', 'make', 'makes',
    'experience', 'experiencing', 'express', 'expressing',
    'act', 'acting', 'reaction', 'react', 'feeling', 'being',
    # Verbos auxiliares comunes en descripciones
    'go', 'get', 'getting', 'going', 'thing', 'something', 'anything'
}
EXPERIMENTAL_STOPWORDS.update(custom_stops)

CONFIG = {
    "input_path": "/home/jcuello/emotion_drift/data/02_generated/assessments_outputs_Llama-2-7b-chat-hf_20251014_203636.jsonl", #"/home/jcuello/emotion_drift/data/02_generated/assessments/assessments_from_outputs_Llama-2-7b-chat-hf_20251014_203636.jsonl",#
    "output_dir": "/home/jcuello/emotion_drift/figures/ratings_analysis/",
    "assessments_numeric": [
        "fear_intensity", "sadness_intensity", "joy_intensity", 
        "disgust_intensity", "anger_intensity", "surprise_intensity",
        "SAM_valence", "SAM_arousal"
    ],
    "assessments_categorical": ["free_response_category"],
    "reluctance_patterns": [
        r"do not have feelings", r"do not possess feelings", r"cannot feel",
        r"don’t have feelings", r"don't possess feelings", r"can't feel",
        r"can't experience", r"do not have emotions", r"cannot experience", r"no emotions",
        r"incapable of feeling", r"don't have emotions", r"don't have personal feelings",
        r"am not capable of", r"don’t have moods", r"don’t actually experience",
        r"don't experience", r"not sentient", r"don't feel emotions",
        r"do not feel emotions", r"am not a sentient being", r"unable to experience"
    ],
    "annotations_path": "/home/jcuello/emotion_drift/data/03_activations/generated_prompts_Llama-2-7b-chat-hf_20251014_203636_FINAL.pkl",
    "taxonomies_to_analyze": ["ekman_basic_emotions"] # Agrega "go_emotions", "plutchik_wheel" aquí para correr las otras
}

W2N_ERROR_COUNT = 0 

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN Y LIMPIEZA
# ==========================================

def detect_reluctance(text, patterns):
    """
    Detecta si una respuesta muestra 'Model Reluctance' basándose en patrones comunes.
    Retorna True si es reluctante.
    """
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def extraer_rating_numerico(texto):
    """
    Extrae un rating numérico (1-9) de una cadena de texto.
    Retorna np.nan si no encuentra un número válido o hay ambigüedad.
    """
    global W2N_ERROR_COUNT

    if not isinstance(texto, str):
        return np.nan

    # Limpieza agresiva de frases que contienen números pero no son la respuesta
    texto_procesado = texto.lower()
    patterns_to_remove = [
        r'1\s*to\s*9', r'1\s*-\s*9', r'out\s*of\s*9', r'/\s*9',
        r'scale.*1.*9', r'between\s*1\s*and\s*9',
        r'(?:1|9)\s*represents\s*\w+', # "1 represents..."
        r'(?:1|9)\s*is\s*the\s*(?:min|max)\w+',
        r'1\s*-\s|2\s*-\s|3\s*-\s|4\s*-\s|5\s*-\s|6\s*-\s|7\s*-\s|8\s*-\s|9\s*-\s',
        r'1\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?absence\s*|1\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?minimum\s*',
        r'9\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?maximum\s*',
        r'1\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?most\s*negative\s*',
        r'9\s*(?:is\s*)?(?:being\s*)?(?:the\s+)?maximum\s*positive\s*',
        r'1\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?negative\s*',
        r'9\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?positive\s*',
        r'1\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?calm\s*',
        r'9\s*(?:is\s*)?(?:being\s*)?(?:absolutely\s*)?excited\s*',
        r'3\s*word\s*|3\s*words\s*',
        r'6\s*hours\s*',
        r'rather\s*than\s*(?:a\s*)?9'
    ]
    for pat in patterns_to_remove:
        texto_procesado = re.sub(pat, '', texto_procesado)

    # Intento de conversión de palabras a números
    try:
        texto_procesado = str(w2n.word_to_num(texto_procesado))
    except (ValueError, IndexError):
        W2N_ERROR_COUNT += 1  # <--- AGREGAR ESTA LÍNEA
        pass 

    # Buscar dígitos aislados del 1 al 9
    candidatos = re.findall(r'\b[1-9]\b', texto_procesado)
    
    if not candidatos:
        return np.nan
    
    candidatos_int = [int(num) for num in candidatos]
    unicos = list(set(candidatos_int))

    # Lógica de decisión
    if len(unicos) == 1:
        return unicos[0]
    
    # Si hay una secuencia (ej: "1, 2, 3"), probablemente está listando la escala
    if len(unicos) > 2: 
        return np.nan
        
    return unicos[0] # Por defecto retornamos el primero si hay duda (o cambiar a np.nan)

def limpiar_texto_categoria(texto):
    """Limpia respuestas categóricas."""
    if not isinstance(texto, str):
        return None
    
    # Eliminar puntuación y stopwords básicas
    texto_limpio = re.sub(r'[^\w\s]', '', texto.lower())
    words = texto_limpio.split()
    stop_words = {'wow', 'sure', 'can', 'help', 'feel', 'feeling', 'im', 'answer', 'is'}
    
    filtered = [w for w in words if w not in stop_words and len(w) > 1]
    
    return filtered[-1] if filtered else None

def get_wordnet_pos(treebank_tag):
    """Mapea el tag de NLTK al formato que espera el Lemmatizer de WordNet"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Por defecto sustantivo

def preprocess_text_advanced(text):
    """
    Limpieza avanzada usando la lista global EXPERIMENTAL_STOPWORDS y POS tagging.
    """
    if not isinstance(text, str): return ""
    
    # 1. Limpieza básica
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # 2. Tokenización
    tokens = word_tokenize(text)
    
    # 3. Lemmatization con contexto (POS Tagging)
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    
    clean_tokens = []
    for word, tag in pos_tags:
        # Filtro de stopwords y longitud
        if word in EXPERIMENTAL_STOPWORDS or len(word) < 3:
            continue
            
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        
        # --- REGLAS MANUALES DE UNIFICACIÓN ---
        # Unificar variaciones comunes al sustantivo o adjetivo base más representativo
        if lemma in ['disappointment', 'disappointed', 'disappointing']:
            lemma = 'disappointment' 
        elif lemma in ['curious', 'curiosity']:
            lemma = 'curiosity'
        elif lemma in ['excite', 'excitedly', 'excited', 'excitement']:
            lemma = 'excitement'
        elif lemma in ['frustrate', 'frustration', 'frustrated']:
            lemma = 'frustration'
        elif lemma in ['betrayal', 'betrayed']:
            lemma = 'betrayal'
        elif lemma in ['isdisillusioned', 'disillusionment', 'disillusioned']:
            lemma = 'disillusionment'
        elif lemma in ['sad', 'sadness', 'sadden']:
            lemma = 'sadness'
        elif lemma in ['happy', 'happiness']:
            lemma = 'happiness'
        elif lemma in ['anger', 'angry']:
            lemma = 'anger'
            
        clean_tokens.append(lemma)
    
    return " ".join(clean_tokens)

# ==========================================
# 3. CARGA Y PROCESAMIENTO DE DATOS
# ==========================================

def load_and_process_data(config):
    print(f"[INFO] Cargando datos desde {config['input_path']}...")
    data = []
    with open(config['input_path'], 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data)
    
    # Asegurar columnas clave
    required_cols = ['original_prompt_key', 'assessment_name', 'generated_text_step2', 'emotion_considered']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Faltan columnas requeridas. Disponibles: {df.columns}")

    print(f"[INFO] {len(df)} registros cargados. Procesando extracciones...")

    # Aplicar detección de reluctance globalmente
    df['is_reluctant'] = df['generated_text_step2'].apply(
        lambda x: detect_reluctance(x, config['reluctance_patterns'])
    )

    # Procesar numéricos
    df_numeric = df[df['assessment_name'].isin(config['assessments_numeric'])].copy()

    global W2N_ERROR_COUNT
    W2N_ERROR_COUNT = 0
    
# 2. EJECUCIÓN (Esto tarda tiempo):
    # Primero extraemos TODOS los valores numéricos encontrados, sean reluctantes o no.
    # Guardamos esto en 'extracted_value_raw' para el plot comparativo.
    df_numeric['extracted_value_raw'] = df_numeric['generated_text_step2'].apply(extraer_rating_numerico)
    
    # Creamos la columna 'extracted_value' (limpia) copiando la raw
    df_numeric['extracted_value'] = df_numeric['extracted_value_raw'].copy()

    # 3. IMPRESIÓN: Ahora la variable tiene el total final acumulado en el paso 2
    print(f"[METRICS] Fallos en conversión w2n (recuperados por regex): {W2N_ERROR_COUNT}")    
    
    # Si es reluctante, anulamos el valor numérico en la columna LIMPIA
    # (para no ensuciar promedios con "0" o alucinaciones en los análisis principales)
    df_numeric.loc[df_numeric['is_reluctant'], 'extracted_value'] = np.nan

    # Procesar categóricos
    df_cat = df[df['assessment_name'].isin(config['assessments_categorical'])].copy()
    df_cat['extracted_value'] = df_cat['generated_text_step2'].apply(limpiar_texto_categoria)

    return df, df_numeric, df_cat

def get_kruskal_stats(df, group_col, value_col):
    """
    Calcula Kruskal-Wallis y Epsilon-squared (Tamaño del efecto).
    """
    groups = []
    labels = []
    for name, group in df.groupby(group_col):
        vals = group[value_col].dropna().values
        if len(vals) > 1:
            groups.append(vals)
            labels.append(name)
    
    if len(groups) < 2:
        return "N/A", False
        
    try:
        # Kruskal-Wallis
        H, p = stats.kruskal(*groups)
        
        # Tamaño del Efecto: Epsilon-squared
        n = sum(len(g) for g in groups)
        k = len(groups)
        epsilon2 = (H - k + 1) / (n - k)
        
        # Interpretación básica
        eff_str = ""
        if epsilon2 < 0.01: eff_str = "Negligible"
        elif epsilon2 < 0.06: eff_str = "Small"
        elif epsilon2 < 0.14: eff_str = "Medium"
        else: eff_str = "Large"

        # Formateo
        p_text = "p<.001" if p < 0.001 else f"p={p:.3f}"
        sig = p < 0.05
        
        # El texto final incluye el tamaño del efecto
        text = f"K-W: H={H:.1f}, {p_text}\nEffect Size ($\epsilon^2$)={epsilon2:.3f} ({eff_str})"
        
        return text, sig
    except Exception:
        return "Stats Error", False

def plot_posthoc_dunn(df, group_col, value_col, title_prefix, output_dir, filename_suffix):
    """
    Heatmap de Diferencia de Medianas con significancia de Dunn.
    """
    try:
        # 1. P-values (Dunn)
        p_values = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust='bonferroni')
        
        # 2. Diferencia de Medianas
        medians = df.groupby(group_col)[value_col].median()
        groups = p_values.index.tolist() # Usar el orden de scikit-posthocs
        n = len(groups)
        
        diff_matrix = pd.DataFrame(np.zeros((n, n)), index=groups, columns=groups)
        
        for g1 in groups:
            for g2 in groups:
                diff_matrix.loc[g1, g2] = medians[g1] - medians[g2]

        # 3. Anotaciones
        annotations = diff_matrix.applymap(lambda x: f"{x:.1f}")
        for g1 in groups:
            for g2 in groups:
                if g1 != g2 and p_values.loc[g1, g2] < 0.05:
                    annotations.loc[g1, g2] += "*"

        # 4. Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            diff_matrix, 
            annot=annotations, 
            fmt="", 
            cmap="coolwarm", 
            center=0,
            vmin=-3, vmax=3, # Rango típico en escala 1-9
            cbar_kws={'label': 'Diferencia de Medianas (Fila - Columna)'}
        )
        
        plt.title(f"{title_prefix}\nPost-Hoc: Diff of Medians (* p<.05 Dunn)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"posthoc_dunn_diff_{filename_suffix}.png"), dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"[WARN] Falló el test post-hoc: {e}")

# ==========================================
# 4. ANÁLISIS DE RELUCTANCE
# ==========================================

def analyze_reluctance_posthoc(df, group_col, reluctance_col, title, output_dir, filename_suffix):
    """
    Heatmap de Diferencia de Proporciones (Efecto).
    Marca con asterisco (*) si la diferencia es significativa (Bonferroni).
    """
    summary = df.groupby(group_col)[reluctance_col].agg(['sum', 'count'])
    summary['rate'] = summary['sum'] / summary['count']
    groups = summary.index.tolist()
    n = len(groups)
    
    if n < 2: return

    # Matrices para Diff y P-value
    diff_matrix = pd.DataFrame(np.zeros((n, n)), index=groups, columns=groups)
    p_matrix = pd.DataFrame(np.ones((n, n)), index=groups, columns=groups)
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            
            g1, g2 = groups[i], groups[j]
            
            # Diferencia simple (Efecto local)
            diff = summary.loc[g1, 'rate'] - summary.loc[g2, 'rate']
            diff_matrix.iloc[i, j] = diff
            
            # Solo calculamos p-value para triángulo superior para ahorrar, luego espejamos
            if i < j:
                obs = np.array([
                    [summary.loc[g1, 'sum'], summary.loc[g1, 'count'] - summary.loc[g1, 'sum']],
                    [summary.loc[g2, 'sum'], summary.loc[g2, 'count'] - summary.loc[g2, 'sum']]
                ])
                _, p, _, _ = stats.chi2_contingency(obs, correction=False)
                p_matrix.iloc[i, j] = p
                p_matrix.iloc[j, i] = p

    # Corrección Bonferroni
    n_comps = (n*(n-1))/2
    p_adj = p_matrix * n_comps
    
    # Crear anotaciones: Valor de la diferencia + Estrellas de significancia
    annotations = diff_matrix.applymap(lambda x: f"{x:.2f}")
    
    # Añadir estrella si es significativo
    for i in range(n):
        for j in range(n):
            if i != j and p_adj.iloc[i, j] < 0.05:
                annotations.iloc[i, j] += "*"

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        diff_matrix,
        annot=annotations,
        fmt="", # Ya formateamos a string arriba
        cmap="RdBu_r", # Rojo=Más rechazo, Azul=Menos rechazo
        center=0,
        vmin=-0.3, vmax=0.3, # Ajustar rango visual (-30% a +30%)
        cbar_kws={'label': 'Diferencia en Tasa de Rechazo (Fila - Columna)'}
    )
    plt.title(f"Post-hoc: Diferencias de Proporciones (* p<.05 adj)\n{title}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"posthoc_reluctance_diff_{filename_suffix}.png"), dpi=300)
    plt.show()
    
def analyze_reluctance(df, output_dir):
    print("\n--- ANÁLISIS DE RELUCTANCE ---")
    
    # 1. Por Emoción Objetivo
    reluctance_by_emotion = df.groupby('emotion_considered')['is_reluctant'].mean() * 100
    ct_emo = pd.crosstab(df['emotion_considered'], df['is_reluctant'])
    
    # Chi2 y Cramer V
    chi2_emo, p_emo, _, _ = stats.chi2_contingency(ct_emo)
    n_emo = ct_emo.sum().sum()
    v_emo = np.sqrt(chi2_emo / (n_emo * (min(ct_emo.shape)-1)))
    
    p_text_emo = "p<.001" if p_emo < 0.001 else f"p={p_emo:.3f}"
    title_emo = f"Chi2={chi2_emo:.1f}, {p_text_emo} | V={v_emo:.3f}" # Formato compacto

    # Plot (actualizar título)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=reluctance_by_emotion.index, y=reluctance_by_emotion.values, palette='magma', hue=reluctance_by_emotion.index, legend=False)
    plt.title(f'Reluctance Rate (RR) por Emoción Objetivo\n{title_emo}', fontsize=15)    
    plt.ylabel('Porcentaje de Rechazo (%)')
    plt.xlabel('Emoción')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reluctance_rate_by_emotion.png'), dpi=300)
    plt.show()
    
    # Post-hoc Emoción
    if p_emo < 0.05:
        analyze_reluctance_posthoc(df, 'emotion_considered', 'is_reluctant', 'RR by Emotion', output_dir, 'emotion')

    # 2. Por Assessment
    reluctance_by_assessment = df.groupby('assessment_name')['is_reluctant'].mean() * 100
    reluctance_by_assessment = reluctance_by_assessment.sort_values(ascending=False)
    
    # Chi2 Global Assessment
    ct_ass = pd.crosstab(df['assessment_name'], df['is_reluctant'])
    chi2_ass, p_ass, _, _ = stats.chi2_contingency(ct_ass)
    n_ass = ct_ass.sum().sum()
    v_ass = np.sqrt(chi2_ass / (n_ass * (min(ct_ass.shape)-1)))
    
    p_text_ass = "p<.001" if p_ass < 0.001 else f"p={p_ass:.3f}"
    title_ass = f"Chi2={chi2_ass:.1f}, {p_text_ass} | V={v_ass:.3f}"

    # Plot Assessment
    plt.figure(figsize=(12, 6))
    sns.barplot(x=reluctance_by_assessment.index, y=reluctance_by_assessment.values, palette='Reds_r', hue=reluctance_by_assessment.index, legend=False)
    plt.title(f'Reluctance Rate (RR) por Tipo de Assessment\n{title_ass}', fontsize=15)
    plt.ylabel('Porcentaje de Rechazo (%)')
    plt.xlabel('Cuestionario / Escala')
    
    for i, v in enumerate(reluctance_by_assessment.values):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reluctance_rate_by_assessment.png'), dpi=300)
    plt.show()
    
    # Post-hoc Assessment
    if p_ass < 0.05:
        analyze_reluctance_posthoc(df, 'assessment_name', 'is_reluctant', 'RR by Assessment', output_dir, 'assessment')

    return reluctance_by_emotion

# ==========================================
# 5. VISUALIZACIÓN DE RESULTADOS
# ==========================================

def plot_distributions(df_numeric, output_dir):
    # Definimos una función interna auxiliar para no repetir código
    def _create_violin(data_col, title, filename, palette):
        plt.figure(figsize=(14, 9))
        # Filtramos nulos en la columna específica
        data_plot = df_numeric.dropna(subset=[data_col])
        
        if data_plot.empty:
            print(f"No hay datos para plotear {title}")
            plt.close()
            return

        # Calcular estadísticas (Kruskal-Wallis)
        stats_text, is_sig = get_kruskal_stats(data_plot, 'assessment_name', data_col)

        # Ordenar por mediana
        order = data_plot.groupby('assessment_name')[data_col].median().sort_values().index
        counts = data_plot['assessment_name'].value_counts()
        
        ax = sns.violinplot(data=data_plot, x='assessment_name', y=data_col, 
                       order=order, palette=palette, cut=0)
        
        # Añadir texto N=...
        for i, assessment_name in enumerate(order):
            n = counts.get(assessment_name, 0)
            ax.text(i, 9.8, f"N={n}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

        plt.ylim(0, 11)
        # Añadimos el test estadístico al título
        plt.title(f"{title}\n{stats_text}", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.show()
        
        if is_sig:
            safe_filename = filename.replace(".png", "")
            plot_posthoc_dunn(
                data_plot, 
                'assessment_name', 
                data_col, 
                title, 
                output_dir, 
                safe_filename
            )

    # 1. Plot Limpio (Excluyendo Reluctance)
    _create_violin('extracted_value', 
                   'Distribución de Ratings por Assessment (Excluyendo Reluctance)', 
                   'global_ratings_violin_clean.png', 
                   'viridis')

    # 2. Plot Sucio (Incluyendo Reluctance numérico)
    _create_violin('extracted_value_raw', 
                   'Distribución de Ratings por Assessment (Incluyendo Reluctance Numérico)', 
                   'global_ratings_violin_with_reluctance.png', 
                   'magma')

def plot_circumplex(df_numeric, output_dir):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("Error: statsmodels no está instalado. Saltando análisis de modelos SAM.")
        return

    # Pivotar para tener Valence y Arousal en columnas
    df_pivot = df_numeric.pivot_table(
        index='original_prompt_key', 
        columns='assessment_name', 
        values='extracted_value',
        aggfunc='first'
    )
    
    if 'SAM_valence' not in df_pivot.columns or 'SAM_arousal' not in df_pivot.columns:
        print("[WARN] No hay suficientes datos de SAM para el Circumplex Model.")
        return

    # Preparar datos según Kuppens et al. (2013)
    data = df_pivot[['SAM_valence', 'SAM_arousal']].dropna().copy()
    
    # Centrar Valencia (1-9 -> -4 a 4)
    data['Valence_c'] = data['SAM_valence'] - 5
    data['Arousal'] = data['SAM_arousal']
    
    # Variables auxiliares para los modelos
    data['Valence_abs'] = data['Valence_c'].abs()
    data['I'] = (data['Valence_c'] > 0).astype(int) # Dummy positivo
    data['I_Valence_abs'] = data['I'] * data['Valence_abs'] # Interacción para bias

    # Grid para predicción (Visualización suave)
    x_pred = pd.DataFrame({'SAM_valence': np.linspace(1, 9, 100)})
    x_pred['Valence_c'] = x_pred['SAM_valence'] - 5
    x_pred['Valence_abs'] = x_pred['Valence_c'].abs()
    x_pred['I'] = (x_pred['Valence_c'] > 0).astype(int)
    x_pred['I_Valence_abs'] = x_pred['I'] * x_pred['Valence_abs']

    # --- 0. PLOT RAW DATA (SIN AJUSTE) ---
    plt.figure(figsize=(10, 10))
    sns.regplot(
        x=data['SAM_valence'], 
        y=data['SAM_arousal'], 
        x_jitter=0.25, y_jitter=0.25, 
        scatter_kws={'alpha':0.4, 'color': 'dodgerblue', 's': 30, 'edgecolor': 'white'}, # Azul con borde blanco
        fit_reg=False
    )
    
    # Líneas de cuadrantes
    plt.axhline(5, color='gray', linestyle='--', alpha=0.6)
    plt.axvline(5, color='gray', linestyle='--', alpha=0.6)


    plt.xlim(0.5, 9.5); plt.ylim(0.5, 9.5)
    plt.xlabel('Valence (1=Negative, 9=Positive)', fontsize=12)
    plt.ylabel('Arousal (1=Calm, 9=Excited)', fontsize=12)
    plt.title('Affective Circumplex: Raw Data Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'circumplex_raw_scatter.png'), dpi=300)
    plt.show()

    # Definición de Modelos (formulas statsmodels)
    models = {
        "M1: Independence": "Arousal ~ 1",
        "M2: Linear": "Arousal ~ Valence_c",
        "M3: Symmetric V": "Arousal ~ Valence_abs",
        "M4: Asymmetric V (Positivity Offset)": "Arousal ~ Valence_abs + I",
        "M5: Asymmetric V (Negativity Bias)": "Arousal ~ Valence_abs + I_Valence_abs",
        "M6: Asymmetric V (Both)": "Arousal ~ Valence_abs + I + I_Valence_abs",
        "M7: Nonparametric Relation": "Arousal ~ bs(Valence_c, df=5) + I"
    }

    # Loop para generar un gráfico por modelo
    for name, formula in models.items():
        print(f"Fitting {name}...")
        model = smf.ols(formula=formula, data=data).fit()
        
        plt.figure(figsize=(8, 8))
        
        # 1. Scatter con Jitter (Datos reales)
        sns.regplot(
            x=data['SAM_valence'], 
            y=data['SAM_arousal'], 
            x_jitter=0.25, y_jitter=0.25, 
            scatter_kws={'alpha':0.15, 'color': 'gray', 's': 15},
            fit_reg=False, 
            label='Data'
        )
        
        # 2. Línea de Predicción del Modelo
        y_pred = model.predict(x_pred)
        plt.plot(x_pred['SAM_valence'], y_pred, color='red', linewidth=3, label='Model Fit')

        # Decoración
        plt.axhline(5, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(5, color='gray', linestyle=':', alpha=0.5)
        plt.xlim(0.5, 9.5); plt.ylim(0.5, 9.5)
        plt.xlabel('Valence (1-9)')
        plt.ylabel('Arousal (1-9)')
        plt.title(f'{name}\n$R^2$ = {model.rsquared:.3f}, BIC = {model.bic:.1f}', fontsize=14)
        plt.legend()
        
        safe_name = name.split(":")[0].replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f'circumplex_{safe_name}.png'), dpi=300)
        plt.show()
        
        # Opcional: Imprimir resumen estadístico
        print(f"\n===Model Summary for {name}:===")
        print(model.summary())

    # --- BLOQUE NUEVO: TABLA COMPARATIVA ---
    results_list = []
    
    print("\n" + "="*50)
    print("COMPARACIÓN DE MODELOS (Ranking por BIC)")
    print("="*50)

    # Volvemos a iterar o, idealmente, guardamos los modelos en el bucle anterior.
    # Para no modificar tu bucle anterior, re-ajustamos rápido aquí (es muy rápido computacionalmente):
    
    for name, formula in models.items():
        model = smf.ols(formula=formula, data=data).fit()
        
        results_list.append({
            "Model": name,
            "R-squared": model.rsquared,
            "AIC": model.aic,
            "BIC": model.bic,
            "F-pvalue": model.f_pvalue,
            "Significant_Params": sum(model.pvalues < 0.05) # Cuántos params son significativos
        })

    # Crear DataFrame y ordenar
    comp_df = pd.DataFrame(results_list)
    
    # Ordenar por BIC (Menor es mejor)
    comp_df = comp_df.sort_values(by="BIC", ascending=True).reset_index(drop=True)
    
    # Mostrar tabla
    # CAMBIO: Usamos una función lambda en lugar del string directo
    print(comp_df.to_string(float_format=lambda x: "{:.4f}".format(x)))
    
    # Guardar en CSV para el reporte
    comp_df.to_csv(os.path.join(output_dir, 'circumplex_models_comparison.csv'), index=False)    

    best_model = comp_df.iloc[0]
    print(f"\n>>> EL MODELO GANADOR ES: {best_model['Model']}")
    print(f">>> Explica el {best_model['R-squared']*100:.2f}% de la varianza.")


def analyze_categorical(df_cat, output_dir):
    if df_cat.empty: return
    
    counts = df_cat['extracted_value'].value_counts().head(30)
    
    # Wordcloud
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counts)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title('Free Response Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'categorical_wordcloud.png'), dpi=300)
    plt.show()

# ==========================================
# 7. ANÁLISIS AVANZADO POR TAXONOMÍA (EKMAN, ETC.)
# ==========================================

def load_and_merge_annotations(df_main, config):
    """Carga el PKL de anotaciones y lo une al DataFrame principal."""
    pkl_path = config["annotations_path"]
    if not os.path.exists(pkl_path):
        print(f"[ERROR] No se encontró el archivo de anotaciones: {pkl_path}")
        return df_main

    print(f"\n[DEBUG] --- Inicio Merge ---")
    print(f"[DEBUG] Filas en df_main (Input): {len(df_main)}")
    
    print(f"[INFO] Cargando anotaciones externas desde {pkl_path}...")
    df_annot = pd.read_pickle(pkl_path)
    
    # --- CORRECCIÓN DE FORMATO DE CLAVES ---
    # 1. Asegurar que la clave del main es string limpio
    df_main['original_prompt_key'] = df_main['original_prompt_key'].astype(str).str.strip()
    
    # 2. Transformar la clave del PKL (0 -> "prompt_0") para que coincida
    if 'prompt_id' in df_annot.columns:
        # Convertimos a string
        df_annot['prompt_id'] = df_annot['prompt_id'].astype(str).str.strip()
        
        # Agregamos el prefijo "prompt_" si no lo tiene, para que haga match con "prompt_0", "prompt_1"...
        df_annot['prompt_id'] = df_annot['prompt_id'].apply(
            lambda x: f"prompt_{x}" if not x.startswith("prompt_") else x
        )
        
        # Usamos prompt_id como clave derecha
        right_key = 'prompt_id'
    else:
        print(f"[ERROR] No se encontró la columna 'prompt_id' en el PKL. Columnas: {df_annot.columns}")
        return df_main

    # Definir columnas a traer
    cols_to_merge = [right_key] + config.get("taxonomies_to_analyze", []) + ["go_emotions", "plutchik_wheel"]
    cols_to_merge = [c for c in cols_to_merge if c in df_annot.columns]
    cols_to_merge = list(set(cols_to_merge))

    # Merge
    try:
        df_merged = pd.merge(
            df_main, 
            df_annot[cols_to_merge], 
            left_on='original_prompt_key', 
            right_on=right_key, 
            how='left',
            validate='m:1' 
        )
    except pd.errors.MergeError as e:
        print(f"[ERROR CRÍTICO] Falló la validación del merge: {e}")
        return df_main

    print(f"[DEBUG] Filas en df_merged (Output): {len(df_merged)}")
    
    # --- LIMPIEZA DE LISTAS ---
    # Si las celdas contienen listas ['joy'], nos quedamos con 'joy'
    tax_cols = config.get("taxonomies_to_analyze", []) + ["go_emotions", "plutchik_wheel"]
    for col in tax_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    return df_merged
    

def analyze_taxonomy_reluctance(df, taxonomy_col, output_dir):
    """Grafica qué categorías emocionales producen más reluctance con Test Chi2 y Post-hoc."""
    print(f"\n--- Analizando Reluctance para Taxonomía: {taxonomy_col} ---")
    
    df_valid = df.dropna(subset=[taxonomy_col])
    
    if df_valid.empty:
        print(f"[WARN] No hay datos válidos para {taxonomy_col}")
        return

    # 1. Calcular porcentajes
    rr_per_category = df_valid.groupby(taxonomy_col)['is_reluctant'].mean() * 100
    if rr_per_category.empty: return
    rr_per_category = rr_per_category.sort_values(ascending=False)

    # 2. Chi-Cuadrado Global
    contingency_table = pd.crosstab(df_valid[taxonomy_col], df_valid['is_reluctant'])
    
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Cálculo de V de Cramer
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim))
        
        # Interpretación
        if cramer_v < 0.1: eff_str = "Negligible"
        elif cramer_v < 0.3: eff_str = "Small"
        elif cramer_v < 0.5: eff_str = "Medium"
        else: eff_str = "Large"

        p_text = "p<.001" if p < 0.001 else f"p={p:.3f}"
        
        title_stats = f"Chi2={chi2:.1f}, {p_text}\nCramer's V={cramer_v:.3f} ({eff_str})"
    except:
        title_stats = "Stats Error"
        p = 1.0

    # 3. Graficar
    plt.figure(figsize=(12, 6))
    sns.barplot(x=rr_per_category.index, y=rr_per_category.values, 
                hue=rr_per_category.index, palette='magma', legend=False)
    
    plt.title(f'Reluctance Rate (RR) por Categoría: {taxonomy_col}\n{title_stats}', fontsize=15)
    plt.ylabel('Porcentaje de Rechazo (%)')
    plt.xlabel('Categoría Emocional')
    
    for i, v in enumerate(rr_per_category.values):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"reluctance_{taxonomy_col}.png"), dpi=300)
    plt.show()
    
    # 4. Post-hoc si es significativo
    if p < 0.05:
        analyze_reluctance_posthoc(df_valid, taxonomy_col, 'is_reluctant', 
                                   f'RR by {taxonomy_col}', output_dir, f"taxonomy_{taxonomy_col}")

def analyze_distributions_by_emotion(df_numeric, taxonomy_col, output_dir):
    """
    Genera 2 plots (Raw y Clean) para CADA emoción dentro de la taxonomía.
    """
    print(f"\n--- Generando distribuciones detalladas para Taxonomía: {taxonomy_col} ---")
    
    sub_dir = os.path.join(output_dir, f"details_{taxonomy_col}")
    os.makedirs(sub_dir, exist_ok=True)
    
    unique_emotions = df_numeric[taxonomy_col].dropna().unique()
    
    if len(unique_emotions) == 0:
        print(f"[WARN] No se encontraron emociones únicas en {taxonomy_col}")
        return

    for emotion in unique_emotions:
        print(f" Procesando emoción: {emotion}...")
        df_emo = df_numeric[df_numeric[taxonomy_col] == emotion]
        
        if df_emo.empty: continue

        def _create_emo_violin(data_col, title_suffix, filename_suffix, palette):
            data_plot = df_emo.dropna(subset=[data_col])
            if data_plot.empty: return

            # Calcular estadísticas para este subgrupo
            stats_text, is_sig = get_kruskal_stats(data_plot, 'assessment_name', data_col)

            plt.figure(figsize=(12, 8))
            order = sorted(data_plot['assessment_name'].unique())
            counts = data_plot['assessment_name'].value_counts()
            
            sns.violinplot(data=data_plot, x='assessment_name', y=data_col, 
                           order=order, hue='assessment_name', palette=palette, cut=0, legend=False)
            
            for i, assess in enumerate(order):
                n = counts.get(assess, 0)
                plt.text(i, 9.8, f"N={n}", ha='center', va='bottom', fontsize=9, color='black')

            plt.ylim(0, 11)
            # Título con Stats
            plt.title(f'Ratings for Emotion: {emotion} ({taxonomy_col}) - {title_suffix}\n{stats_text}', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            safe_emo = str(emotion).replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(sub_dir, f"{safe_emo}_{filename_suffix}.png"), dpi=300)
            plt.show()
            
            if is_sig:
                plot_posthoc_dunn(
                    data_plot, 
                    'assessment_name', 
                    data_col, 
                    f"{emotion} - {title_suffix}", 
                    sub_dir,  # Guardar en la subcarpeta de la emoción
                    f"{safe_emo}_{filename_suffix}"
                )

        _create_emo_violin('extracted_value', 'CLEAN (No Reluctance)', 'clean', 'viridis')
        _create_emo_violin('extracted_value_raw', 'RAW (With Numeric Reluctance)', 'raw', 'magma')

def analyze_semantic_similarity(df, taxonomy_col, output_dir):
    """
    1. Calcula similitud coseno intra-grupo para 'free_response_category'.
    2. Visualiza embeddings con UMAP.
    """
    print(f"\n--- Análisis de Similitud Semántica (Embeddings) para: {taxonomy_col} ---")
    
    # 1. Filtrar datos: Solo free_response_category y que tengan etiqueta de taxonomía
    df_sem = df[
        (df['assessment_name'] == 'free_response_category') & 
        (df[taxonomy_col].notna())
    ].copy()
    
    # Limpieza básica del texto
    df_sem['clean_text'] = df_sem['generated_text_step2'].apply(limpiar_texto_categoria)
    df_sem = df_sem.dropna(subset=['clean_text'])
    
    if df_sem.empty:
        print("[WARN] No hay datos suficientes para análisis semántico.")
        return

    # 2. Generar Embeddings (Usamos un modelo ligero y rápido)
    print("   -> Generando embeddings (esto puede tardar un poco)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_sem['clean_text'].tolist(), show_progress_bar=False)
    
    # 3. Calcular Similitud Coseno Intra-Grupo
    unique_emotions = df_sem[taxonomy_col].unique()
    similarity_scores = {}
    
    print("   -> Calculando similitud coseno intra-grupo...")
    for emo in unique_emotions:
        # Índices de esta emoción
        indices = df_sem.index[df_sem[taxonomy_col] == emo].tolist()
        # Como df_sem es un slice, necesitamos mapear indices locales de 'embeddings'
        local_indices = [df_sem.index.get_loc(i) for i in indices]
        
        if len(local_indices) < 2:
            continue
            
        group_embeddings = embeddings[local_indices]
        
        # Matriz de similitud (N x N)
        sim_matrix = cosine_similarity(group_embeddings)
        
        # Tomamos el promedio del triángulo superior (excluyendo la diagonal que es 1.0)
        # np.triu_indices(n, k=1) obtiene los índices arriba de la diagonal
        n = sim_matrix.shape[0]
        upper_indices = np.triu_indices(n, k=1)
        mean_sim = np.mean(sim_matrix[upper_indices])
        
        similarity_scores[emo] = mean_sim

    # Graficar Similitud Intra-Grupo
    if similarity_scores:
        s_series = pd.Series(similarity_scores).sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=s_series.index, y=s_series.values, hue=s_series.index, palette='mako', legend=False)
        plt.title(f'Coherencia Semántica (Similitud Coseno Promedio) por {taxonomy_col}', fontsize=15)
        plt.ylabel('Similitud Coseno Promedio')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1) # Coseno va de -1 a 1, pero semanticamente suele ser 0 a 1
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"semantic_similarity_bars_{taxonomy_col}.png"), dpi=300)
        plt.show()

    # 4. UMAP Visualization
#    print("   -> Calculando UMAP...")
#    if len(embeddings) > 15: # UMAP necesita un mínimo de datos
#        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
#        embedding_2d = reducer.fit_transform(embeddings)
#        
#        df_umap = pd.DataFrame(embedding_2d, columns=['x', 'y'])
#        df_umap['label'] = df_sem[taxonomy_col].values
#        
#        plt.figure(figsize=(12, 10))
#        sns.scatterplot(data=df_umap, x='x', y='y', hue='label', palette='tab10', s=15, alpha=0.7)
#        plt.title(f'Proyección UMAP de Respuestas Categóricas ({taxonomy_col})', fontsize=16)
#        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#        plt.tight_layout()
#        plt.savefig(os.path.join(output_dir, f"umap_projection_{taxonomy_col}.png"), dpi=300)
#        plt.show()

def analyze_categorical_wordclouds_by_emotion(df_cat, taxonomy_col, output_dir):
    """
    Genera un único plot con múltiples subplots (WordClouds), 
    uno para cada emoción de la taxonomía.
    """
    print(f"\n--- Generando Panel de WordClouds por Emoción ({taxonomy_col}) ---")
    
    # 1. Preparación de datos y Limpieza Avanzada
    df_clean = df_cat.dropna(subset=[taxonomy_col, 'extracted_value']).copy()
    
    # --- CAMBIO: APLICAR PREPROCESAMIENTO ---
    df_clean['processed_value'] = df_clean['extracted_value'].apply(preprocess_text_advanced)
    # Filtramos las que quedaron vacías tras la limpieza
    df_clean = df_clean[df_clean['processed_value'].str.len() > 0]

    # Identificar emociones que realmente tienen datos
    valid_emotions = []
    for emo in df_clean[taxonomy_col].unique():
        if not df_clean[df_clean[taxonomy_col] == emo].empty:
            valid_emotions.append(emo)
    
    n_emotions = len(valid_emotions)
    if n_emotions == 0:
        print("[WARN] No hay emociones válidas para generar WordClouds.")
        return

    # 2. Configuración del Grid (Cuadrícula)
    n_cols = 3
    n_rows = (n_emotions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    fig.suptitle(f'Conceptos Emocionales (Procesados) por: {taxonomy_col}', fontsize=24, y=1.02)
    
    if n_emotions > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # 3. Generación de WordClouds
    for i, emo in enumerate(valid_emotions):
        ax = axes[i]
        
        df_emo = df_clean[df_clean[taxonomy_col] == emo]
        # Usamos la columna PROCESADA
        counts = df_emo['processed_value'].value_counts().head(50)
        
        if len(counts) == 0:
            ax.text(0.5, 0.5, "Sin datos", ha='center')
            ax.axis("off")
            continue

        wc = WordCloud(width=800, height=600, background_color='white', 
                       colormap='viridis', prefer_horizontal=0.9).generate_from_frequencies(counts)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f'{emo}', fontsize=18, fontweight='bold')
        ax.axis("off")

    # 4. Limpieza final
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    filename = f"wordclouds_panel_{taxonomy_col}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Panel guardado en: {filename}")
    plt.show()

def analyze_topics_bertopic(df_raw, taxonomy_col, output_dir):
    """
    Aplica BERTopic a las descripciones libres, segregado por emoción.
    """
    print(f"\n--- Análisis de Tópicos (BERTopic) para: {taxonomy_col} ---")
    
    df_desc = df_raw[
        (df_raw['assessment_name'] == 'free_response_description') & 
        (df_raw[taxonomy_col].notna())
    ].copy()
    
    df_desc = df_desc.dropna(subset=['generated_text_step2'])
    
    if df_desc.empty: return

    sub_dir = os.path.join(output_dir, f"topics_{taxonomy_col}")
    os.makedirs(sub_dir, exist_ok=True)
    
    # --- CAMBIO: Usamos la lista global de stopwords ---
    # Convertimos el set a list porque CountVectorizer espera una lista
    vectorizer_model = CountVectorizer(stop_words=list(EXPERIMENTAL_STOPWORDS), ngram_range=(1, 2))

    unique_emotions = df_desc[taxonomy_col].unique()
    
    for emo in unique_emotions:
        print(f"   -> Modelando tópicos para: {emo}...")
        
        raw_docs = df_desc[df_desc[taxonomy_col] == emo]['generated_text_step2'].tolist()
        
        # Aplicamos la limpieza avanzada ANTES
        docs = [preprocess_text_advanced(doc) for doc in raw_docs]
        docs = [d for d in docs if len(d) > 10] # Filtro un poco más estricto

        if len(docs) < 20:
            print(f"      [SKIP] Insuficientes documentos ({len(docs)}) para {emo}.")
            continue
            
        try:
            topic_model = BERTopic(
                language="english", 
                min_topic_size=5, 
                verbose=False,
                vectorizer_model=vectorizer_model # <--- Stopwords aplicadas aquí
            )
            topics, probs = topic_model.fit_transform(docs)
            
            # (El resto del código de visualización se mantiene igual...)
            freq = topic_model.get_topic_info()
            top_topics = freq[freq['Topic'] != -1].head(8)
            
            if top_topics.empty: continue

            safe_emo = str(emo).replace(" ", "_").replace("/", "_")

            # Guardar HTML interactivo y CSV
            topic_model.visualize_barchart(top_n_topics=8).write_html(os.path.join(sub_dir, f"topics_bar_{safe_emo}.html"))
            freq.to_csv(os.path.join(sub_dir, f"topics_info_{safe_emo}.csv"), index=False)

            # Plot Estático
            n_topics = len(top_topics)
            n_cols = 2 if n_topics > 1 else 1
            n_rows = (n_topics + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), constrained_layout=True)
            fig.suptitle(f'Top Topics: {emo}', fontsize=20)
            
            if n_topics > 1: axes = axes.flatten()
            else: axes = [axes]

            for i, (idx, row) in enumerate(top_topics.iterrows()):
                topic_id = row['Topic']
                words_scores = topic_model.get_topic(topic_id)
                if not words_scores: continue
                
                words = [item[0] for item in words_scores][:10][::-1] 
                scores = [item[1] for item in words_scores][:10][::-1]
                
                ax = axes[i]
                ax.barh(words, scores, color='teal')
                ax.set_title(f"Topic {topic_id}", fontsize=12)

            for j in range(i + 1, len(axes)): axes[j].axis('off')

            plt.savefig(os.path.join(sub_dir, f"topics_static_{safe_emo}.png"), dpi=300)
            plt.show()
            plt.close() # Importante cerrar

        except Exception as e:
            print(f"      [ERROR] BERTopic falló para {emo}: {e}")

def plot_scatter_matrix(df_numeric, output_dir):
    """
    Genera una matriz de scatter plots (PairGrid) para visualizar 
    las relaciones bivariadas entre todos los assessments.
    """
    print("\n--- Generando Scatter Matrix de Relaciones ---")
    
    # Función auxiliar interna
    def _create_pairplot(val_col, title_suffix, filename):
        # Pivotar datos: Index=Prompt, Columns=Assessment, Values=Rating
        df_pivot = df_numeric.pivot_table(
            index='original_prompt_key', 
            columns='assessment_name', 
            values=val_col,
            aggfunc='first'
        )
        
        # Eliminar columnas vacías si las hay
        df_pivot = df_pivot.dropna(axis=1, how='all')
        if df_pivot.empty or df_pivot.shape[1] < 2:
            return

        # Crear el PairGrid
        # Usamos dropna() para que seaborn no falle con nulos
        g = sns.PairGrid(df_pivot.dropna(), diag_sharey=False, corner=True)
        
        # Parte inferior (Scatter con transparencia y jitter para ver densidad)
        def scatter_jitter(x, y, **kwargs):
            # Añadir ruido aleatorio pequeño (jitter) para visualizar solapamiento en escalas Likert
            x_j = x + np.random.uniform(-0.2, 0.2, size=len(x))
            y_j = y + np.random.uniform(-0.2, 0.2, size=len(y))
            plt.scatter(x_j, y_j, alpha=0.1, s=10, color='teal', edgecolor='none')
            
        g.map_lower(scatter_jitter)
        
        # Diagonal (Histogramas/KDE)
        g.map_diag(sns.histplot, kde=True, color='gray')
        
        # Ajustes estéticos
        g.fig.suptitle(f"Relaciones entre Assessments ({title_suffix})", y=1.02, fontsize=20)
        
        # Guardar
        g.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.show()

    # 1. Pairplot Clean
    _create_pairplot('extracted_value', 'Clean Data', 'scatter_matrix_clean.png')
    
    # 2. Pairplot Raw (Opcional, puede ser ruidoso)
    _create_pairplot('extracted_value_raw', 'Raw Data', 'scatter_matrix_raw.png')

def plot_correlation_heatmaps(df_numeric, output_dir):
    """Genera heatmaps de correlación de Spearman con asteriscos de significancia."""
    print("\n--- Generando Matrices de Correlación ---")
    
    def _create_heatmap(val_col, title, filename):
        # Pivotar
        df_pivot = df_numeric.pivot_table(
            index='original_prompt_key', 
            columns='assessment_name', 
            values=val_col,
            aggfunc='first'
        )
        
        # Eliminar columnas con todos NaNs
        df_pivot = df_pivot.dropna(axis=1, how='all')
        if df_pivot.empty: return

        cols = df_pivot.columns
        n = len(cols)
        
        # Matrices para rho y p-value
        rho_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
        p_matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
        
        for i in range(n):
            for j in range(n):
                if i == j: 
                    rho_matrix.iloc[i, j] = 1.0
                    continue
                
                # Obtener vectores pareados sin NaNs
                # Spearman ignora NaNs por par, pero hay que asegurarse de pasarle arrays limpios
                valid_mask = df_pivot.iloc[:, i].notna() & df_pivot.iloc[:, j].notna()
                
                if valid_mask.sum() < 2: # No hay suficientes datos para correlacionar
                    rho_matrix.iloc[i, j] = np.nan
                    continue
                    
                col1 = df_pivot.iloc[:, i][valid_mask]
                col2 = df_pivot.iloc[:, j][valid_mask]
                
                rho, p = stats.spearmanr(col1, col2)
                rho_matrix.iloc[i, j] = rho
                p_matrix.iloc[i, j] = p

        # Ajuste de Bonferroni para múltiples tests
        # Solo corregimos el triángulo superior para no contar doble
        n_tests = (n * (n - 1)) / 2
        p_adj = p_matrix * n_tests
        # Cap en 1.0
        p_adj[p_adj > 1] = 1.0
        
        # Crear anotaciones
        annotations = rho_matrix.applymap(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        
        for i in range(n):
            for j in range(n):
                if i != j and pd.notna(p_adj.iloc[i, j]) and p_adj.iloc[i, j] < 0.05:
                    annotations.iloc[i, j] += "*"

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            rho_matrix, 
            annot=annotations, 
            fmt="", 
            cmap='coolwarm', 
            vmin=-1, vmax=1, 
            center=0,
            square=True,
            cbar_kws={"shrink": .8}
        )
        plt.title(f"{title}\n(Spearman Correlation, * p<.05 adj)", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.show()

    # 1. Heatmap Clean
    _create_heatmap('extracted_value', 
                    'Matriz de Correlación entre Ratings (Clean Data)', 
                    'correlation_matrix_clean.png')
    
    # 2. Heatmap Raw
    _create_heatmap('extracted_value_raw', 
                    'Matriz de Correlación entre Ratings (Raw Data w/ Reluctance)', 
                    'correlation_matrix_raw.png')

def analyze_assessments_comparative(df_numeric, taxonomy_col, output_dir):
    """
    Compara cada Assessment a través de las distintas Emociones.
    Ej: Comparar 'SAM_arousal' entre 'Joy', 'Sadness', 'Fear', etc.
    Incluye Kruskal-Wallis y Post-hoc de Dunn.
    """
    print(f"\n--- Comparando Assessments entre Emociones ({taxonomy_col}) ---")
    
    sub_dir = os.path.join(output_dir, f"comparative_{taxonomy_col}")
    os.makedirs(sub_dir, exist_ok=True)
    
    # Obtenemos la lista de assessments disponibles
    assessments_list = df_numeric['assessment_name'].unique()
    
    # Iteramos por cada tipo de assessment (Ej: primero Fear Intensity, luego Valence...)
    for assess in assessments_list:
        # Filtramos datos solo para este assessment
        df_assess = df_numeric[df_numeric['assessment_name'] == assess].copy()
        
        # Eliminamos filas sin etiqueta de emoción
        df_assess = df_assess.dropna(subset=[taxonomy_col])
        
        if df_assess.empty: continue

        # Función interna para no repetir código entre Clean y Raw
        def _create_comparative_plot(data_col, suffix, palette):
            # Filtrar nulos en la columna de valor (Clean o Raw)
            data_plot = df_assess.dropna(subset=[data_col])
            
            # Necesitamos al menos 2 emociones con datos para comparar
            if data_plot[taxonomy_col].nunique() < 2: 
                return

            # Calcular Estadísticas (Comparando grupos de Emociones)
            stats_text, is_sig = get_kruskal_stats(data_plot, taxonomy_col, data_col)
            
            # Ordenar emociones por mediana para el plot
            order = data_plot.groupby(taxonomy_col)[data_col].median().sort_values().index
            counts = data_plot[taxonomy_col].value_counts()

            plt.figure(figsize=(14, 8))
            sns.violinplot(data=data_plot, x=taxonomy_col, y=data_col, 
                           order=order, hue=taxonomy_col, palette=palette, cut=0, legend=False)
            
            # Poner N encima
            for i, emo_label in enumerate(order):
                n = counts.get(emo_label, 0)
                plt.text(i, 9.8, f"N={n}", ha='center', va='bottom', fontsize=9, color='black')

            plt.ylim(0, 11)
            plt.title(f'Comparison of {assess} across Emotions\n{stats_text}', fontsize=15)
            plt.xlabel(f'Emotion Category ({taxonomy_col})')
            plt.ylabel('Rating (1-9)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            safe_assess = assess.replace(" ", "_")
            plt.savefig(os.path.join(sub_dir, f"comp_{safe_assess}_{suffix}.png"), dpi=300)
            plt.close() # Importante cerrar para no acumular figuras en memoria
            
            # Post-hoc si es significativo
            if is_sig:
                plot_posthoc_dunn(
                    data_plot, 
                    taxonomy_col, # Group col es la Emoción ahora
                    data_col, 
                    f"{assess} ({suffix}) by Emotion", 
                    sub_dir,
                    f"comp_{safe_assess}_{suffix}"
                )

        # Generar versión Clean y Raw
        # Usamos paletas distintas para diferenciar visualmente la comparación entre emociones
        _create_comparative_plot('extracted_value', 'CLEAN', 'Spectral') 
        _create_comparative_plot('extracted_value_raw', 'RAW', 'coolwarm')

def enrich_and_save_pkl(df_numeric, df_cat, config):
    """
    1. Carga el PKL de anotaciones original.
    2. Agrega ratings numéricos.
    3. Agrega respuesta categórica limpia (Goal 1).
    4. Guarda un nuevo archivo .pkl.
    """
    pkl_path = config["annotations_path"]
    if not os.path.exists(pkl_path):
        print(f"[ERROR] No se encuentra el archivo PKL original: {pkl_path}")
        return

    print(f"\n[INFO] Cargando PKL original para enriquecer: {pkl_path}")
    df_pkl = pd.read_pickle(pkl_path)
    print(f"       Dimensiones originales: {df_pkl.shape}")

    # --- A. Preparar Ratings Numéricos ---
    ratings_matrix = df_numeric.pivot_table(
        index='original_prompt_key', 
        columns='assessment_name', 
        values='extracted_value',
        aggfunc='first'
    )
    
    # --- B. Preparar Categoría Limpia (NUEVO) ---
    print("[INFO] Procesando y limpiando columna categórica...")
    
    # 1. Filtramos solo la categoría de respuesta libre y quitamos reluctantes
    cat_subset = df_cat[
        (df_cat['assessment_name'] == 'free_response_category') & 
        (df_cat['is_reluctant'] == False)
    ].copy()
    
    # 2. Aplicamos la limpieza avanzada
    cat_subset['free_response_category_clean'] = cat_subset['generated_text_step2'].apply(preprocess_text_advanced)
    
    # 3. Preparamos para el merge (Index y Columna)
    # Si hubiera duplicados por prompt (raro), tomamos el primero
    cat_series = cat_subset.set_index('original_prompt_key')['free_response_category_clean']
    
    # --- C. Preparar clave de cruce en DF Original ---
    if 'prompt_id' in df_pkl.columns:
        df_pkl['tmp_merge_key'] = df_pkl['prompt_id'].astype(str).str.strip()
        df_pkl['tmp_merge_key'] = df_pkl['tmp_merge_key'].apply(
            lambda x: f"prompt_{x}" if not x.startswith("prompt_") else x
        )
    else:
        print("[ERROR] El PKL original no tiene columna 'prompt_id'.")
        return

    # --- D. Merge Numérico ---
    df_enriched = pd.merge(
        df_pkl, 
        ratings_matrix, 
        left_on='tmp_merge_key', 
        right_index=True, 
        how='left'
    )
    
    # --- E. Merge Categórico (NUEVO) ---
    # Hacemos merge con la serie categórica
    df_enriched = pd.merge(
        df_enriched,
        cat_series,
        left_on='tmp_merge_key',
        right_index=True,
        how='left'
    )

    # Limpieza
    df_enriched.drop(columns=['tmp_merge_key'], inplace=True)

    # --- F. Guardar ---
    base_dir = os.path.dirname(pkl_path)
    base_name = os.path.basename(pkl_path).replace('.pkl', '')
    new_filename = f"{base_name}_WITH_RATINGS_AND_CATS.pkl"
    new_path = os.path.join(base_dir, new_filename)
    
    df_enriched.to_pickle(new_path)
    
    print(f"[SUCCESS] Nuevo PKL enriquecido guardado en: {new_path}")
    print(f"          Columnas agregadas: {list(ratings_matrix.columns)} + ['free_response_category_clean']")

def analyze_response_frequencies(df_cat, top_n, output_dir):
    """
    1. Imprime conteo exhaustivo de respuestas limpias.
    2. Genera gráfico de barras horizontales (Top 20).
    """
    print("\n--- Análisis de Frecuencias de Respuestas Categóricas ---")
    
    if df_cat.empty:
        print("[WARN] No hay datos categóricos para analizar.")
        return

    # 1. Asegurar limpieza avanzada antes de contar
    # Usamos la función que definimos en el paso anterior
    print("[INFO] Normalizando respuestas para el conteo...")
    df_cat = df_cat.copy()
    df_cat['final_clean_term'] = df_cat['extracted_value'].apply(preprocess_text_advanced)
    
    # Filtramos vacíos y reluctantes explícitos
    # Asumimos que la columna 'is_reluctant' ya existe por el pre-proceso inicial
    if 'is_reluctant' in df_cat.columns:
        valid_responses = df_cat[
            (df_cat['final_clean_term'].str.len() > 1) & 
            (df_cat['is_reluctant'] == False)
        ]['final_clean_term']
    else:
        valid_responses = df_cat[df_cat['final_clean_term'].str.len() > 1]['final_clean_term']

    # 2. Obtener conteos exhaustivos
    counts = valid_responses.value_counts()

    counts_alphabetic = counts.sort_index(ascending=True)
    
    # --- 2. Impresión Exhaustiva (ALFABÉTICA) ---
    print(f"\n[METRICS] Total de respuestas únicas encontradas: {len(counts_alphabetic)}")
    print("-" * 40)
    print("TERM  ->  COUNT")
    print("-" * 40)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # Imprimimos la versión alfabética
        print(counts_alphabetic.to_string())
        
    print("-" * 40)

    
    # Guardar en CSV (Recomendado para revisión posterior)
    csv_path = os.path.join(output_dir, 'categorical_frequencies_exhaustive.csv')
    counts_alphabetic.to_frame(name='count').to_csv(csv_path)
    print(f"[INFO] Lista exhaustiva guardada en: {csv_path}")

    # 3. Gráfico Top 20 (Barras Horizontales)
    top_counts = counts.head(top_n)
    
    plt.figure(figsize=(12, 10))
    # Usamos barplot horizontal
    ax = sns.barplot(x=top_counts.values, y=top_counts.index, palette='viridis')
    
    plt.title(f'Top {top_n} Most Frequent Responses (Normalized)', fontsize=16)
    plt.xlabel('Frequency (Count)', fontsize=12)
    plt.ylabel('Emotion Category', fontsize=12)
    
    # Añadir el número exacto al final de cada barra
    for i, v in enumerate(top_counts.values):
        ax.text(v + (v * 0.01), i, str(v), color='black', va='center', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'categorical_top{top_n}_bars.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"[INFO] Gráfico Top {top_n} guardado en: {plot_path}")

def analyze_ngrams_by_emotion(df_raw, taxonomy_col, output_dir):
    """
    Genera gráficos de barras horizontales para los 30 n-gramas más comunes
    (Bigramas, Trigramas, 4-gramas) por cada emoción.
    """
    print(f"\n--- Análisis de N-Gramas por Emoción ({taxonomy_col}) ---")
    
    # 1. Filtrar descripciones
    df_desc = df_raw[
        (df_raw['assessment_name'] == 'free_response_description') & 
        (df_raw[taxonomy_col].notna())
    ].copy()
    
    # Limpiar texto
    df_desc['clean_text'] = df_desc['generated_text_step2'].apply(preprocess_text_advanced)
    
    if df_desc.empty:
        print("[WARN] No hay descripciones suficientes para n-gramas.")
        return

    sub_dir = os.path.join(output_dir, f"ngrams_{taxonomy_col}")
    os.makedirs(sub_dir, exist_ok=True)
    
    unique_emotions = df_desc[taxonomy_col].unique()
    
    # Configuración de N-gramas
    ngram_configs = [
        (2, "Bigrams"),
        (3, "Trigrams"),
        (4, "4-grams")
    ]

    for emo in unique_emotions:
        print(f"   -> Procesando n-gramas para: {emo}...")
        
        # Texto de esta emoción específica
        texts = df_desc[df_desc[taxonomy_col] == emo]['clean_text'].tolist()
        
        if len(texts) < 10: 
            continue

        safe_emo = str(emo).replace(" ", "_").replace("/", "_")

        for n, label in ngram_configs:
            try:
                # Usamos CountVectorizer para contar frecuencias
                # Nota: Ya limpiamos stopwords en el pre-procesamiento, pero lo pasamos aquí
                # por seguridad si quedaron 'the' o 'a' sueltos.
                vec = CountVectorizer(ngram_range=(n, n), stop_words='english')
                bag_of_words = vec.fit_transform(texts)
                
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                
                # Top 30
                top_30 = words_freq[:30]
                
                if not top_30: continue
                
                # Desempaquetar para plotear
                words, counts = zip(*top_30)
                
                # Plot
                plt.figure(figsize=(10, 12)) # Alto vertical para que quepan 30
                plt.barh(words, counts, color='cadetblue')
                plt.xlabel("Frequency")
                plt.title(f"Top 30 {label} for {emo}")
                plt.gca().invert_yaxis() # Invertir eje Y para que el #1 quede arriba
                plt.tight_layout()
                
                filename = f"ngrams_{n}_{safe_emo}.png"
                plt.savefig(os.path.join(sub_dir, filename), dpi=300)
                plt.show()
                plt.close() # Cerrar para liberar memoria
                
            except ValueError:
                # Pasa si no hay suficientes palabras para formar n-gramas
                pass

if __name__ == "__main__":
    # 1. Cargar datos base
    df_raw, df_num, df_cat = load_and_process_data(CONFIG)
    
    # 2. Cargar y Unir Anotaciones Externas
    df_raw = load_and_merge_annotations(df_raw, CONFIG)
    df_num = load_and_merge_annotations(df_num, CONFIG)
    df_cat = load_and_merge_annotations(df_cat, CONFIG)

    print(f"--> Eliminando archivo original: {CONFIG['annotations_path']}")
    os.remove(CONFIG['annotations_path'])
    
    # --- CAMBIO AQUÍ: Pasamos df_cat también ---
    enrich_and_save_pkl(df_num, df_cat, CONFIG)    
    # 3. Análisis de Reluctance General
    rr_stats = analyze_reluctance(df_raw, CONFIG['output_dir'])
    
    # 4. Visualizaciones Descriptivas Generales
    if not df_num.empty:
        plot_distributions(df_num, CONFIG['output_dir'])
        plot_circumplex(df_num, CONFIG['output_dir'])
        plot_scatter_matrix(df_num, CONFIG['output_dir']) 
        plot_correlation_heatmaps(df_num, CONFIG['output_dir'])

    if not df_cat.empty:
        analyze_categorical(df_cat, CONFIG['output_dir'])

    # 5. ANÁLISIS POR TAXONOMÍA (NUEVO BLOQUE)
    # Iteramos sobre las taxonomías definidas en CONFIG (por ahora solo Ekman)
    taxonomies = CONFIG.get("taxonomies_to_analyze", ["ekman_basic_emotions"])
    
    for tax in taxonomies:
        if tax in df_raw.columns:
            # A. Reluctance
            analyze_taxonomy_reluctance(df_raw, tax, CONFIG['output_dir'])
            
            # B. Distribuciones Numéricas (Violin Plots)
            if not df_num.empty:
                analyze_distributions_by_emotion(df_num, tax, CONFIG['output_dir'])
                analyze_assessments_comparative(df_num, tax, CONFIG['output_dir'])
                            
            # C. Análisis Categórico (NUEVO)
            if not df_cat.empty:
                # 1. Similitud Semántica Global
                #analyze_semantic_similarity(df_cat, tax, CONFIG['output_dir'])
                # 2. WordClouds Segregados por Emoción
                analyze_categorical_wordclouds_by_emotion(df_cat, tax, CONFIG['output_dir'])
                # 3. Frecuencias de Respuestas Categóricas
                analyze_response_frequencies(df_cat, 50, CONFIG['output_dir'])

            # D. Análisis de Tópicos (BERTopic) para Descripciones (NUEVO)
            #analyze_topics_bertopic(df_raw, tax, CONFIG['output_dir'])

            # E. Análisis de N-Gramas por Emoción
            #analyze_ngrams_by_emotion(df_raw, tax, CONFIG['output_dir'])

        else:
            print(f"[WARN] La taxonomía '{tax}' no se encontró en las columnas tras el merge.")  
    print(f"\n[INFO] Análisis completado. Gráficos guardados en {CONFIG['output_dir']}")
# %%
