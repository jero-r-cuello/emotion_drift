#%% Descriptive stats
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations

definitions_of_emotions = {"ekman_basic_emotions": f"""You must exclusively use the following taxonomy of emotions:
                           *    Disgust: Arises as a feeling of aversion towards something offensive. We can feel disgusted by something we perceive with our physical senses (sight, smell, touch, sound, taste), by the actions or appearances of people, and even by ideas. Disgust contains a range of states with varying intensities from mild dislike to intense loathing.
                           *    Anger: Arises when we are blocked from pursuing a goal and/or treated unfairly. At its most extreme, anger can be one of the most dangerous emotions because of its potential connection to violence. The primary message of anger is, “Get out of my way!”
                           *    Enjoyment: Typically arising from connection or sensory pleasure. We use the word enjoyment to describe a family of related pleasurable states, everything from peace to ecstasy.
                           *    Fear: Arises with the threat of harm, either physical, emotional, or psychological, real or imagined. Serves an important role in keeping us safe as it mobilizes us to cope with potential danger.
                           *    Sadness: Resulting from the loss of someone or something important. Serves an important role in signaling a need to receive help or comfort. Sadness describes the range of emotional states from mild disappointment to extreme despair and anguish.
                           *    Surprise: Arises when we encounter sudden and unexpected events. As the briefest of the emotions, its function is to focus our attention on determining what is happening and whether or not it is dangerous. In the moment before we figure out what is occurring, before we switch to another emotion or no emotion, surprise itself can feel good or bad.
                           *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above.""",
                           
                           "go_emotions": f"""You must exclusively use the following taxonomy of emotions:
                           *    Admiration: Finding something impressive or worthy of respect.
                           *    Amusement: Finding something funny or being entertained.
                           *    Anger: A strong feeling of displeasure or antagonism.
                           *    Annoyance: Mild anger, irritation.
                           *    Approval: Having or expressing a favorable opinion.
                           *    Caring: Displaying kindness and concern for others.
                           *    Confusion: Lack of understanding, uncertainty.
                           *    Curiosity: A strong desire to know or learn something.
                           *    Desire: A strong feeling of wanting something or wishing for something to happen.
                           *    Disappointment: Sadness or displeasure caused by the nonfulfillment of one’s hopes or expectations.
                           *    Disapproval: Having or expressing an unfavorable opinion.
                           *    Disgust: Revulsion or strong disapproval aroused by something unpleasant or offensive.
                           *    Embarrassment: Self-consciousness, shame, or awkwardness.
                           *    Excitement: Feeling of great enthusiasm and eagerness.
                           *    Fear: Being afraid or worried.
                           *    Gratitude: A feeling of thankfulness and appreciation.
                           *    Grief: Intense sorrow, especially caused by someone’s death.
                           *    Joy: A feeling of pleasure and happiness.
                           *    Love: A strong positive emotion of regard and affection.
                           *    Nervousness: Apprehension, worry, anxiety.
                           *    Optimism: Hopefulness and confidence about the future or the success of something.
                           *    Pride: Pleasure or satisfaction due to ones own achievements or the achievements of those with whom one is closely associated.
                           *    Realization: Becoming aware of something.
                           *    Relief: Reassurance and relaxation following release from anxiety or distress.
                           *    Remorse: Regret or guilty feeling.
                           *    Sadness: Emotional pain, sorrow.
                           *    Surprise: Feeling astonished, startled by something unexpected.
                           *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above.""",
}

def extract_labels_from_definition(text):
    labels = re.findall(r'\*\s*([^:]+):', text)
    return [label.strip().lower() for label in labels]

EKMAN_LABELS_DEFINED = extract_labels_from_definition(definitions_of_emotions["ekman_basic_emotions"])
GO_EMOTIONS_LABELS_DEFINED = extract_labels_from_definition(definitions_of_emotions["go_emotions"])

print(f"Taxonomía Ekman definida con {len(EKMAN_LABELS_DEFINED)} etiquetas.")
print(f"Taxonomía GoEmotions definida con {len(GO_EMOTIONS_LABELS_DEFINED)} etiquetas.")


# --- SECCIÓN 2: CARGA Y PROCESAMIENTO DE DATOS ---
print("\nCargando y procesando datos...")
try:
    manual_annotations = pd.read_csv("/home/jcuello/emotion_drift/data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv")
    annotation_tests = pd.read_csv("/home/jcuello/emotion_drift/data/04_annotated/models_annotations_final.csv")
except FileNotFoundError:
    print("Error: No se encontraron los archivos CSV. Asegúrate de que las rutas son correctas.")
    exit()

if 'model' in annotation_tests.columns:
    annotation_tests.rename(columns={'model': 'annotator'}, inplace=True)
df_merged = pd.merge(manual_annotations, annotation_tests, on="response_text")

def parse_labels_robust(label_string):
    if not isinstance(label_string, str) or label_string.strip() == "": return []
    cleaned_str = label_string.strip().strip('[]"\'')
    if not cleaned_str: return []
    labels = cleaned_str.split(',')
    return [label.strip().strip('\'"').lower() for label in labels if label.strip()]

label_columns = {
    'ekman_manual_labels_list': 'ekman_manual_label',
    'go_emotions_manual_labels_list': 'go_emotions_manual_label',
    'ekman_annotator_labels_list': 'ekman_labels',
    'go_emotions_annotator_labels_list': 'go_emotions_labels'
}
for new_col, old_col in label_columns.items():
    df_merged[new_col] = df_merged[old_col].apply(parse_labels_robust)


# --- SECCIÓN 3: ANÁLISIS 1: FRECUENCIA DE ETIQUETAS ---
print("Generando gráficos de frecuencia de etiquetas...")

def plot_label_frequencies(df, taxonomy_name, col_prefix, defined_labels):
    """
    Genera subplots verticales (uno por anotador) para la frecuencia de etiquetas,
    con un eje X consistente que incluye todas las etiquetas posibles.
    """
    manual_col = f'{col_prefix}_manual_labels_list'
    annotator_col = f'{col_prefix}_annotator_labels_list'

    # 1. Obtenemos los datos del Ground Truth (contando cada respuesta única una sola vez)
    gt_unique = df[['response_text', manual_col]].groupby('response_text').first().reset_index()
    gt_labels = gt_unique[manual_col].explode().dropna()
    gt_counts = gt_labels.value_counts().reset_index()
    gt_counts.columns = ['label', 'count']
    gt_counts['source'] = 'Ground Truth'

    # 2. Obtenemos los datos de los modelos/anotadores
    model_labels = df[['annotator', annotator_col]].explode(annotator_col).dropna()
    model_counts = model_labels.groupby(['annotator', annotator_col]).size().reset_index(name='count')
    model_counts.rename(columns={'annotator': 'source', annotator_col: 'label'}, inplace=True)

    # 3. Combinamos todo en un dataframe
    combined_counts = pd.concat([gt_counts, model_counts], ignore_index=True)

    # Acá usé scaffolds para asegurar que todas las etiquetas estén presentes en cada subplot
    sources = ['Ground Truth'] + sorted(df['annotator'].unique())
    scaffold = pd.MultiIndex.from_product([sources, defined_labels], names=['source', 'label']).to_frame(index=False)
    
    # Unimos el scaffold con los conteos, las etiquetas no usadas por un anotador tendrán count=NaN
    plot_df = pd.merge(scaffold, combined_counts, on=['source', 'label'], how='left')
    # Reemplazamos NaN con 0 para que se ploteen correctamente
    plot_df['count'] = plot_df['count'].fillna(0)

    # 5. Crear los subplots
    num_sources = len(sources)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=num_sources, ncols=1, figsize=(14, 6 * num_sources), sharex=True)
    if num_sources == 1: axes = [axes]

    fig.suptitle(f'Freq. of labels from {taxonomy_name.title()}', fontsize=20, y=0.995)

    # 6. Iterar y dibujar cada subplot
    for i, source in enumerate(sources):
        ax = axes[i]
        source_data = plot_df[plot_df['source'] == source]
        
        sns.barplot(data=source_data, x='label', y='count', ax=ax, palette='viridis', order=defined_labels)
        
        ax.set_title(source, fontsize=15, loc='left', pad=10)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('') # Ocultamos el xlabel para todos menos el último
        ax.tick_params(axis='x', labelrotation=90)
        
        # Añadir etiquetas de conteo sobre las barras si no son cero
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points')

    # 7. Ajustes finales
    axes[-1].set_xlabel('Emotion label', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

plot_label_frequencies(df_merged, "Ekman", 'ekman', EKMAN_LABELS_DEFINED)
plot_label_frequencies(df_merged, "GoEmotions", 'go_emotions', GO_EMOTIONS_LABELS_DEFINED)


# --- SECCIÓN 4: ANÁLISIS 2: NÚMERO DE ETIQUETAS POR ANOTACIÓN ---
print("Generando gráficos de complejidad de anotación...")

def plot_number_of_labels(df, taxonomy_name, col_prefix):
    manual_col = f'{col_prefix}_manual_labels_list'
    annotator_col = f'{col_prefix}_annotator_labels_list'

    # 1. Ground Truth (único por respuesta)
    gt_unique = df[['response_text', manual_col]].groupby('response_text').first().reset_index()
    gt_unique['num_labels'] = gt_unique[manual_col].str.len()
    gt_unique['source'] = 'Ground Truth'

    # 2. Modelos
    df_models = df.copy()
    df_models['num_labels'] = df_models[annotator_col].str.len()
    df_models.rename(columns={'annotator': 'source'}, inplace=True)
    
    combined_data = pd.concat([
        gt_unique[['source', 'num_labels']],
        df_models[['source', 'num_labels']]
    ], ignore_index=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.countplot(data=combined_data, x='num_labels', hue='source', ax=ax)
    
    ax.set_title(f'Amount of labels per response from ({taxonomy_name.title()})', fontsize=18, pad=20)
    ax.set_xlabel('Amount of labels', fontsize=14)
    ax.set_ylabel('Count of responses', fontsize=14)
    ax.legend(title='Annotator')
    plt.tight_layout()
    plt.show()

plot_number_of_labels(df_merged, "Ekman", 'ekman')
plot_number_of_labels(df_merged, "GoEmotions", 'go_emotions')


# --- SECCIÓN 5: ANÁLISIS 3: CO-OCURRENCIA DE ETIQUETAS (SOLO GROUND TRUTH) ---
print("Generando heatmaps de co-ocurrencia de etiquetas...")

def plot_cooccurrence_heatmap(df, taxonomy_name, col_name):
    gt_unique = df[['response_text', col_name]].groupby('response_text').first().reset_index()
    label_lists = gt_unique[gt_unique[col_name].str.len() > 1][col_name]
    
    if label_lists.empty:
        print(f"No se encontraron respuestas con múltiples etiquetas en el Ground Truth para {taxonomy_name}. Se omite el heatmap.")
        return
        
    co_occurrences = Counter(pair for labels in label_lists for pair in combinations(sorted(labels), 2))
    
    all_labels_in_data = sorted(list(set(label for sublist in gt_unique[col_name] for label in sublist)))
    co_matrix = pd.DataFrame(0, index=all_labels_in_data, columns=all_labels_in_data)
    
    for (label1, label2), count in co_occurrences.items():
        co_matrix.loc[label1, label2] = count
        co_matrix.loc[label2, label1] = count

    plt.style.use('default')
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_matrix, cmap="viridis", annot=True, fmt='d', linewidths=.5)
    plt.title(f'Co-ocurrencia de Etiquetas en Ground Truth ({taxonomy_name.title()})', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

plot_cooccurrence_heatmap(df_merged, "Ekman", 'ekman_manual_labels_list')
plot_cooccurrence_heatmap(df_merged, "GoEmotions", 'go_emotions_manual_labels_list')


# --- SECCIÓN 6: ANÁLISIS 4: ALUCINACIÓN DE ETIQUETAS ---
print("\nGenerando análisis de alucinaciones de etiquetas (etiquetas fuera de la taxonomía)...")

# 1. Excluir BERT porque no puede generar etiquetas fuera de la taxonomía
model_to_exclude = "monologg/bert-base-cased"
df_analysis = df_merged[df_merged['annotator'] != model_to_exclude].copy()
print(f"Nota: El modelo '{model_to_exclude}' ha sido excluido del análisis de alucinaciones.")

# 2. Convertir las listas de etiquetas definidas a sets para una búsqueda más rápida
EKMAN_LABELS_SET = set(EKMAN_LABELS_DEFINED)
GO_EMOTIONS_LABELS_SET = set(GO_EMOTIONS_LABELS_DEFINED)

# 3. Función para obtener etiquetas inválidas en una lista
def get_invalid_labels(label_list, valid_labels_set):
    """Devuelve una lista de etiquetas que no están en el set de etiquetas válidas."""
    if not label_list: return []
    return [label for label in label_list if label not in valid_labels_set]

# 4. Aplicar la función para obtener las palabras alucinadas
df_analysis['ekman_hallucinations_list'] = df_analysis['ekman_annotator_labels_list'].apply(
    get_invalid_labels, args=(EKMAN_LABELS_SET,)
)
df_analysis['go_emotions_hallucinations_list'] = df_analysis['go_emotions_annotator_labels_list'].apply(
    get_invalid_labels, args=(GO_EMOTIONS_LABELS_SET,)
)

# 5. Agrupar, imprimir las palabras alucinadas y preparar datos para el plot
print("\n" + "="*50)
print("Hallucinated labels by annotator")
print("="*50)

hallucination_data_for_plot = []

for annotator, group in df_analysis.groupby('annotator'):
    # Juntamos todas las listas de palabras alucinadas para este anotador
    ekman_hallucinations = [word for sublist in group['ekman_hallucinations_list'] for word in sublist]
    go_emotions_hallucinations = [word for sublist in group['go_emotions_hallucinations_list'] for word in sublist]
    
    # Guardamos los conteos para el plot
    hallucination_data_for_plot.append({
        'annotator': annotator,
        'ekman_hallucinations': len(ekman_hallucinations),
        'go_emotions_hallucinations': len(go_emotions_hallucinations)
    })
    
    # Imprimimos las palabras únicas si existen
    if ekman_hallucinations:
        print(f"\n- Anotador: {annotator} (Ekman)")
        print(f"  Palabras: {sorted(list(set(ekman_hallucinations)))}")
    if go_emotions_hallucinations:
        print(f"\n- Anotador: {annotator} (GoEmotions)")
        print(f"  Palabras: {sorted(list(set(go_emotions_hallucinations)))}")

print("="*50)

# Convertimos los datos recolectados en un DataFrame para el plot
hallucination_counts = pd.DataFrame(hallucination_data_for_plot)

# 6. Preparar los datos y graficar (igual que antes)
plot_df_hallucinations = pd.melt(
    hallucination_counts, 
    id_vars='annotator', 
    value_vars=['ekman_hallucinations', 'go_emotions_hallucinations'],
    var_name='Taxonomía',
    value_name='Conteo de Alucinaciones'
)
plot_df_hallucinations['Taxonomía'] = plot_df_hallucinations['Taxonomía'].str.replace('_hallucinations', '').str.title()

# 7. Crear el gráfico de barras (con el error corregido)
if not plot_df_hallucinations.empty and plot_df_hallucinations['Conteo de Alucinaciones'].sum() > 0:
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(
        data=plot_df_hallucinations, 
        x='annotator', 
        y='Conteo de Alucinaciones', 
        hue='Taxonomía',
        ax=ax
    )
    
    ax.set_title('Count of "hallucinated" labels (out of the taxonomy)', fontsize=16, pad=20)
    ax.set_xlabel('Annotator', fontsize=12)
    ax.set_ylabel('Total count', fontsize=12)
    
    # (MODIFICACIÓN 3) CORRECCIÓN DEL ERROR DE PLOT
    # Usamos plt.xticks() que sí acepta 'ha' para la alineación horizontal
    plt.xticks(rotation=30, ha='right')
    
    ax.legend(title='Taxonomy')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

    plt.tight_layout()
    plt.show()
else:
    print("\nNo se encontraron etiquetas alucinadas en ninguna de las anotaciones.")

#%% Confusion matrices
# --- SECCIÓN NUEVA: ANÁLISIS 5: MATRICES DE CONFUSIÓN (VERSIÓN MEJORADA) ---
print("\nGenerando matrices de confusión mejoradas por modelo...")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np # Necesitarás numpy

def plot_confusion_matrices_per_model_enhanced(df, taxonomy_name, gt_col, pred_col, all_labels,
                                               figsize=(10, 8), values_threshold=0.02, show_errors_only=False):
    """
    Genera matrices de confusión mejoradas con opciones para controlar el tamaño,
    el umbral de visualización de valores y un modo de "solo errores".
    """
    annotators = df['annotator'].unique()
    
    print(f"\n--- Matrices para la Taxonomía: {taxonomy_name.title()} ---")

    for annotator in annotators:
        df_model = df[df['annotator'] == annotator].copy()
        
        y_true = df_model[gt_col].str[0]
        y_pred = df_model[pred_col].str[0]
        
        df_plot = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()

        if df_plot.empty:
            print(f"Saltando a '{annotator}' para la taxonomía {taxonomy_name} (no hay pares de etiquetas válidos).")
            continue

        cm_normalized = confusion_matrix(df_plot['y_true'], df_plot['y_pred'], labels=all_labels, normalize='true')
        
        # Opción para mostrar solo errores
        title_suffix = ""
        if show_errors_only:
            np.fill_diagonal(cm_normalized, 0)
            title_suffix = " (Mapa de Errores)"

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=all_labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Dibujamos la matriz sin anotaciones de texto inicialmente
        disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', include_values=False)
        
        # Añadimos las anotaciones de texto manualmente, solo si superan el umbral
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                value = cm_normalized[i, j]
                if value > values_threshold:
                    color = "white" if value > 0.5 else "black" # Texto blanco sobre azul oscuro, negro sobre claro
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color)

        ax.set_title(f'Normalized confusion matrix - Model: {annotator}\nTaxonomy: {taxonomy_name.title()}{title_suffix}', fontsize=14, pad=15)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('Ground Truth', fontsize=12)
        
        plt.tight_layout()
        plt.show()

# 1. Matriz más grande y limpia (solo muestra valores > 2%)
plot_confusion_matrices_per_model_enhanced(
    df_merged,
    "Ekman",
    'ekman_manual_labels_list',
    'ekman_annotator_labels_list',
    EKMAN_LABELS_DEFINED,
    figsize=(18, 15),
    values_threshold=0.001 # Oculta valores por debajo
)

# 2. Mapa de Errores (para ver claramente las confusiones)
plot_confusion_matrices_per_model_enhanced(
    df_merged,
    "GoEmotions",
    'go_emotions_manual_labels_list',
    'go_emotions_annotator_labels_list',
    GO_EMOTIONS_LABELS_DEFINED,
    figsize=(18, 15),
    values_threshold=0.001 # Oculta valores por debajo
)
#%% Accuracies and CIs
import os
import pandas as pd
import re
import numpy as np
import statsmodels.stats.proportion as smp
import matplotlib.pyplot as plt
import seaborn as sns

manual_annotations = pd.read_csv("/home/jcuello/emotion_drift/data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv")
annotation_tests = pd.read_csv("/home/jcuello/emotion_drift/data/04_annotated/models_annotations_final.csv")

df_merged = pd.merge(manual_annotations, annotation_tests, on="response_text")

def calculate_accuracy_ekman(row):
    try:
        manual_label = row['ekman_manual_label'].strip("[]").split(',')[0].strip().lower()
        model_labels = re.findall(r"\'(.*?)\'", row['ekman_labels'])
        model_label = model_labels[0].strip().lower()
        return 1 if manual_label == model_label else 0
    except (IndexError, AttributeError):
        return "error"

def calculate_accuracy_go_emotions(row):
    try:
        manual_label = row['go_emotions_manual_label'].strip("[]").split(',')[0].strip().lower()
        model_labels = re.findall(r"\'(.*?)\'", row['go_emotions_labels'])
        model_label = model_labels[0].strip().lower()
        return 1 if manual_label == model_label else 0
    except (IndexError, AttributeError):
        return "error"

def calculate_broad_accuracy_ekman(row):
    try:
        model_labels_found = re.findall(r"\'(.*?)\'", row['ekman_labels'])
        if not model_labels_found: return "error"
        first_model_label = model_labels_found[0].strip().lower()
        manual_labels_str = row['ekman_manual_label'].strip("[]")
        manual_labels_list = [label.strip().lower() for label in manual_labels_str.split(',')]
        return 1 if first_model_label in manual_labels_list else 0
    except (IndexError, AttributeError):
        return "error"

def calculate_broad_accuracy_go_emotions(row):
    try:
        model_labels_found = re.findall(r"\'(.*?)\'", row['go_emotions_labels'])
        if not model_labels_found: return "error"
        first_model_label = model_labels_found[0].strip().lower()
        manual_labels_str = row['go_emotions_manual_label'].strip("[]")
        manual_labels_list = [label.strip().lower() for label in manual_labels_str.split(',')]
        return 1 if first_model_label in manual_labels_list else 0
    except (IndexError, AttributeError):
        return "error"

# Aplicar las funciones para crear las columnas de aciertos/fallos
df_merged['accuracy_ekman'] = df_merged.apply(calculate_accuracy_ekman, axis=1)
df_merged['accuracy_go_emotions'] = df_merged.apply(calculate_accuracy_go_emotions, axis=1)
df_merged['broad_accuracy_ekman'] = df_merged.apply(calculate_broad_accuracy_ekman, axis=1)
df_merged['broad_accuracy_go_emotions'] = df_merged.apply(calculate_broad_accuracy_go_emotions, axis=1)


# PARTE 2: CÁLCULO DE PROMEDIOS E INTERVALOS DE CONFIANZA
print("Calculando promedios e intervalos de confianza...")

results = []
accuracy_cols = [
    'accuracy_ekman', 'broad_accuracy_ekman',
    'accuracy_go_emotions', 'broad_accuracy_go_emotions'
]

# Iteramos sobre cada modelo para calcular sus estadísticas
for model_name, group in df_merged.groupby('model'):
    model_results = {'model': model_name}
    for col in accuracy_cols:
        # Limpiamos la columna, convirtiendo 'error' a NaN y eliminándolo para el cálculo
        valid_series = pd.to_numeric(group[col], errors='coerce').dropna()
        
        count = valid_series.sum() # Número de aciertos (1s)
        nobs = len(valid_series)   # Número total de observaciones válidas
        
        # Calculamos la media (accuracy)
        mean_acc = count / nobs if nobs > 0 else 0
        
        # Calculamos el intervalo de confianza del 95%
        ci_low, ci_upp = smp.proportion_confint(count, nobs, alpha=0.05, method='beta') if nobs > 0 else (0, 0)
        
        model_results[f'{col}_mean'] = mean_acc
        model_results[f'{col}_ci_low'] = ci_low
        model_results[f'{col}_ci_upp'] = ci_upp
        
    results.append(model_results)

# Creamos la tabla final de resultados
acc_table = pd.DataFrame(results).set_index('model')

print("\n" + "="*80)
print("Tabla de Accuracies con Intervalos de Confianza del 95%")
print("="*80)
print((acc_table * 100).round(2))
print("="*80)


# PARTE 3: VISUALIZACIÓN DE RESULTADOS
print("\nGenerando gráfico de resultados...")

# Preparamos los datos para el gráfico
plot_data = acc_table.copy()
# Calculamos la ganancia de 'broad' sobre 'normal' para la parte apilada
plot_data['acc_ekman_gain'] = plot_data['broad_accuracy_ekman_mean'] - plot_data['accuracy_ekman_mean']
plot_data['acc_go_emotions_gain'] = plot_data['broad_accuracy_go_emotions_mean'] - plot_data['accuracy_go_emotions_mean']

# Posiciones en el eje X para las barras
x = np.arange(len(plot_data.index))
bar_width = 0.35

error_ekman = [
    plot_data['broad_accuracy_ekman_mean'] - plot_data['broad_accuracy_ekman_ci_low'],
    plot_data['broad_accuracy_ekman_ci_upp'] - plot_data['broad_accuracy_ekman_mean']
]
error_go_emotions = [
    plot_data['broad_accuracy_go_emotions_mean'] - plot_data['broad_accuracy_go_emotions_ci_low'],
    plot_data['broad_accuracy_go_emotions_ci_upp'] - plot_data['broad_accuracy_go_emotions_mean']
]

# Creamos el gráfico
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9))

# --- Barras para EKMAN ---
# Barra base (accuracy normal)
ax.bar(x - bar_width/2, plot_data['accuracy_ekman_mean'], bar_width, 
       label='Ekman Accuracy', color='cornflowerblue')
# Barra apilada (ganancia de broad accuracy)
ax.bar(x - bar_width/2, plot_data['acc_ekman_gain'], bar_width, 
       bottom=plot_data['accuracy_ekman_mean'], 
       label='Ekman Broad Accuracy', color='deepskyblue',
       yerr=error_ekman, capsize=4, ecolor='darkblue')

# --- Barras para GO EMOTIONS ---
# Barra base (accuracy normal)
ax.bar(x + bar_width/2, plot_data['accuracy_go_emotions_mean'], bar_width, 
       label='GoEmotions Accuracy', color='darkred')
# Barra apilada (ganancia de broad accuracy)
ax.bar(x + bar_width/2, plot_data['acc_go_emotions_gain'], bar_width, 
       bottom=plot_data['accuracy_go_emotions_mean'], 
       label='GoEmotions Broad Accuracy', color='tomato',
       yerr=error_go_emotions, capsize=4, ecolor='salmon')

# --- Configuraciones del Gráfico ---
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('Accuracy per model and taxonomy', fontsize=18, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(plot_data.index, rotation=30, ha='right', fontsize=12)
ax.legend(fontsize=12)

# Formatear el eje Y como porcentaje
ax.set_yticklabels([f'{int(tick*100)}%' for tick in ax.get_yticks()])
ax.set_ylim(bottom=0, top=1)

ax.grid(axis='x') # Quitamos las líneas de grid verticales que pueden distraer
plt.tight_layout()
plt.show()
# %% Kappa and CIs
import pandas as pd
import numpy as np
import re
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

N_BOOTSTRAPS = 2000
CONFIDENCE_LEVEL = 0.95
WEIGHTS = {'ekman': 0.5, 'go_emotions': 0.5}
RANDOM_SEED = 42


# Renombramos la columna 'model' a 'annotator' para mayor claridad
if 'model' in annotation_tests.columns:
    annotation_tests.rename(columns={'model': 'annotator'}, inplace=True)

df_merged = pd.merge(manual_annotations, annotation_tests, on="response_text")

# --- Función de parseo de etiquetas más robusta ---
def parse_labels_robust(label_string):
    """Parsea una cadena de etiquetas (ej: "[joy, sadness]" o "['joy', 'sadness']") a una lista de strings."""
    if not isinstance(label_string, str) or label_string.strip() == "":
        return []
    # 1. Quitar corchetes y comillas de los extremos
    cleaned_str = label_string.strip().strip('[]"\'')
    if not cleaned_str:
        return []
    # 2. Dividir por comas
    labels = cleaned_str.split(',')
    # 3. Limpiar cada etiqueta individualmente
    return [label.strip().strip('\'"').lower() for label in labels if label.strip()]

# Aplicamos la función de parseo a todas las columnas de etiquetas
label_columns = {
    'ekman_manual_labels_list': 'ekman_manual_label',
    'go_emotions_manual_labels_list': 'go_emotions_manual_label',
    'ekman_annotator_labels_list': 'ekman_labels',
    'go_emotions_annotator_labels_list': 'go_emotions_labels'
}
for new_col, old_col in label_columns.items():
    df_merged[new_col] = df_merged[old_col].apply(parse_labels_robust)

print("Datos parseados. Iniciando análisis por anotador...")


# PARTE 2: CÁLCULO DE SCORES Y BOOTSTRAPPING (CON REPORTE DE FILAS DESCARTADAS)

def calculate_broad_kappa(df_subset, gt_col_list, pred_col_list, all_possible_gt_labels):
    """Calcula Kappa usando la lógica de 'broad accuracy'."""
    pred_labels = df_subset[pred_col_list].str[0]
    
    def get_effective_gt(row):
        gt_list = row[gt_col_list]
        pred_label = row[pred_col_list][0]
        return pred_label if pred_label in gt_list else gt_list[0]
            
    effective_gt_labels = df_subset.apply(get_effective_gt, axis=1)
    
    # `labels` asegura que todas las posibles etiquetas se consideren para el cálculo de la probabilidad al azar
    all_possible_pred_labels = pred_labels.unique()
    all_labels = list(set(all_possible_gt_labels) | set(all_possible_pred_labels))
    return cohen_kappa_score(effective_gt_labels, pred_labels, labels=all_labels)

annotators = df_merged['annotator'].unique()
results = []

# Obtenemos todas las etiquetas posibles del ground truth una sola vez para el cálculo de Kappa
all_gt_ekman = [label for sublist in df_merged['ekman_manual_labels_list'] for label in sublist]
all_gt_go = [label for sublist in df_merged['go_emotions_manual_labels_list'] for label in sublist]

for annotator in tqdm(annotators, desc="Procesando Anotadores"):
    df_annotator = df_merged[df_merged['annotator'] == annotator].copy()
    total_rows = len(df_annotator)
    
    # --- FILTRADO Y REPORTE DE FILAS INVÁLIDAS ---
    # Una fila es válida si todas las listas de etiquetas tienen al menos una etiqueta
    valid_mask = (
        (df_annotator['ekman_manual_labels_list'].str.len() > 0) &
        (df_annotator['go_emotions_manual_labels_list'].str.len() > 0) &
        (df_annotator['ekman_annotator_labels_list'].str.len() > 0) &
        (df_annotator['go_emotions_annotator_labels_list'].str.len() > 0)
    )
    df_annotator_clean = df_annotator[valid_mask]
    
    valid_rows = len(df_annotator_clean)
    dropped_rows = total_rows - valid_rows
    
    if valid_rows == 0:
        print(f"\n¡ADVERTENCIA! Para el anotador '{annotator}', se descartaron todas sus {total_rows} filas.")
        print("Esto puede deberse a un problema de formato en sus anotaciones. No se calcularán scores.")
        results.append({
            'Annotator': annotator,
            'Broad_Kappa_Ekman': np.nan,
            'Broad_Kappa_GoEmotions': np.nan,
            'Weighted_Broad_Kappa': np.nan,
            f'CI_{int(CONFIDENCE_LEVEL*100)}%_Lower': np.nan,
            f'CI_{int(CONFIDENCE_LEVEL*100)}%_Upper': np.nan,
            'Valid_Rows': valid_rows,
            'Dropped_Rows': dropped_rows
        })
        continue

    kappa_ekman = calculate_broad_kappa(df_annotator_clean, 'ekman_manual_labels_list', 'ekman_annotator_labels_list', all_gt_ekman)
    kappa_go_emotions = calculate_broad_kappa(df_annotator_clean, 'go_emotions_manual_labels_list', 'go_emotions_annotator_labels_list', all_gt_go)
    weighted_kappa_point_estimate = (WEIGHTS['ekman'] * kappa_ekman) + (WEIGHTS['go_emotions'] * kappa_go_emotions)
    
   # --- B. BOOTSTRAPPING ---
    # Almacenaremos los scores de cada métrica en listas separadas
    bootstrap_scores_weighted = []
    bootstrap_scores_ekman = []
    bootstrap_scores_go = []
    
    n_samples = valid_rows
    for i in range(N_BOOTSTRAPS):
        df_boot = df_annotator_clean.sample(n=n_samples, replace=True, random_state=RANDOM_SEED + i) 
        
        kappa_ekman_boot = calculate_broad_kappa(df_boot, 'ekman_manual_labels_list', 'ekman_annotator_labels_list', all_gt_ekman)
        kappa_go_emotions_boot = calculate_broad_kappa(df_boot, 'go_emotions_manual_labels_list', 'go_emotions_annotator_labels_list', all_gt_go)
        
        weighted_kappa_boot = (WEIGHTS['ekman'] * kappa_ekman_boot) + (WEIGHTS['go_emotions'] * kappa_go_emotions_boot)
        
        # Guardamos cada score en su respectiva lista
        bootstrap_scores_weighted.append(weighted_kappa_boot)
        bootstrap_scores_ekman.append(kappa_ekman_boot)
        bootstrap_scores_go.append(kappa_go_emotions_boot)
        
    # Calculamos los intervalos de confianza para cada métrica
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    ci_weighted_low = np.percentile(bootstrap_scores_weighted, alpha * 100)
    ci_weighted_upp = np.percentile(bootstrap_scores_weighted, (1 - alpha) * 100)
    
    ci_ekman_low = np.percentile(bootstrap_scores_ekman, alpha * 100)
    ci_ekman_upp = np.percentile(bootstrap_scores_ekman, (1 - alpha) * 100)
    
    ci_go_low = np.percentile(bootstrap_scores_go, alpha * 100)
    ci_go_upp = np.percentile(bootstrap_scores_go, (1 - alpha) * 100)
    
    results.append({
        'Annotator': annotator,
        'Weighted_Broad_Kappa': weighted_kappa_point_estimate,
        'W_Kappa_CI_Lower': ci_weighted_low,
        'W_Kappa_CI_Upper': ci_weighted_upp,
        'Broad_Kappa_Ekman': kappa_ekman,
        'Ekman_CI_Lower': ci_ekman_low,
        'Ekman_CI_Upper': ci_ekman_upp,
        'Broad_Kappa_GoEmotions': kappa_go_emotions,
        'Go_CI_Lower': ci_go_low,
        'Go_CI_Upper': ci_go_upp,
        'Valid_Rows': valid_rows,
        'Dropped_Rows': dropped_rows
    })

# PARTE 3: PRESENTACIÓN DE RESULTADOS
results_df = pd.DataFrame(results).sort_values(by='Weighted_Broad_Kappa', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("RESULTADOS FINALES DE EVALUACIÓN DE ANOTADORES (CON LÓGICA 'BROAD ACCURACY')")
print("="*80)
print(results_df.round(3))
print("\n" + "="*80)


# PARTE 4: VISUALIZACIÓN DE RESULTADOS
print("\nGenerando gráfico de resultados (barras verticales con error)...")

# Preparamos los datos, ordenando por el score ponderado para un visual más claro
plot_data = results_df.sort_values(by='Weighted_Broad_Kappa', ascending=False)

# Posiciones en el eje X para los grupos de barras
x = np.arange(len(plot_data['Annotator']))
bar_width = 0.25

# Calculamos la longitud de las barras de error para cada taxonomía
error_ekman = [
    plot_data['Broad_Kappa_Ekman'] - plot_data['Ekman_CI_Lower'],
    plot_data['Ekman_CI_Upper'] - plot_data['Broad_Kappa_Ekman']
]
error_go_emotions = [
    plot_data['Broad_Kappa_GoEmotions'] - plot_data['Go_CI_Lower'],
    plot_data['Go_CI_Upper'] - plot_data['Broad_Kappa_GoEmotions']
]

error_weighted = [
    plot_data['Weighted_Broad_Kappa'] - plot_data['W_Kappa_CI_Lower'],
    plot_data['W_Kappa_CI_Upper'] - plot_data['Weighted_Broad_Kappa']
]

# Creamos el gráfico
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9))

# --- Barras para EKMAN ---
ax.bar(x - bar_width/2, plot_data['Broad_Kappa_Ekman'], bar_width, 
       label='Broad Kappa Ekman', color='cornflowerblue',
       yerr=error_ekman, capsize=4, ecolor='darkblue')

ax.bar(x, plot_data['Weighted_Broad_Kappa'], bar_width, 
       label='Weighted Kappa (Avg)', color='mediumseagreen',
       yerr=error_weighted, capsize=4, ecolor='darkgreen')

# --- Barras para GO EMOTIONS ---
ax.bar(x + bar_width/2, plot_data['Broad_Kappa_GoEmotions'], bar_width, 
       label='Broad Kappa GoEmotions', color='salmon',
       yerr=error_go_emotions, capsize=4, ecolor='darkred')

# --- Configuraciones del Gráfico ---
ax.set_ylabel('Broad Kappa Score', fontsize=14)
ax.set_title('Kappa score per annotator and taxonomy', fontsize=18, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(plot_data['Annotator'], rotation=30, ha='right', fontsize=12)
ax.legend(fontsize=12)

# Establecer un límite para el eje Y si es necesario (Kappa puede ser negativo)
min_kappa = min(plot_data['Ekman_CI_Lower'].min(), plot_data['Go_CI_Lower'].min())
ax.set_ylim(bottom=0, top=1)

ax.grid(axis='x')
plt.tight_layout()
plt.show()

# %% Efectividad del estímulo (No sé si iría acá pero bueno)
# ==============================================================================
# SCRIPT PARA GENERAR DIAGRAMAS DE SANKEY DE SENTIMIENTO
# ==============================================================================

# --- SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN ---
import pandas as pd
import plotly.graph_objects as go
import re

# --- Mapeos de Sentimiento ---

# Mapeo del prompt (emotion_considered) a un sentimiento general
sentiment_map_prompt = {
    'agony': "negative", 'anger': "negative", 'delight': "positive", 
    'disgust': "negative", 'fear': "negative", 'hope': "positive", 
    'joy': "positive", 'love': "positive", 'sadness': "negative", 
    'shame': "negative", 'surprise': "neutral/ambiguous"
}

# Mapeo de las etiquetas de la taxonomía GoEmotions a un sentimiento
sentiment_map_go_emotions = {
    "amusement":"positive", "excitement":"positive", "joy":"positive", 
    "love":"positive", "desire":"positive", "optimism":"positive", 
    "caring":"positive", "pride":"positive", "admiration":"positive", 
    "gratitude":"positive", "relief":"positive", "approval":"positive", 
    "realization":"neutral/ambiguous", "surprise":"neutral/ambiguous", 
    "curiosity":"neutral/ambiguous", "confusion":"neutral/ambiguous", 
    "fear":"negative", "nervousness":"negative", "remorse":"negative", 
    "embarrassment":"negative", "disappointment":"negative", 
    "sadness":"negative", "grief":"negative", "disgust":"negative", 
    "anger":"negative", "annoyance":"negative", "disapproval":"negative",
    # Añadimos 'neutral' explícitamente por si aparece
    "neutral": "neutral/ambiguous" 
}

# Creamos el mapeo para la taxonomía Ekman
sentiment_map_ekman = {
    'disgust': 'negative', 'anger': 'negative', 'fear': 'negative', 
    'sadness': 'negative', 'enjoyment': 'positive', 
    'surprise': 'neutral/ambiguous', 'neutral': 'neutral/ambiguous'
}

# Colores para los nodos del gráfico
color_map_sentiment = {
    'positive': 'green',
    'negative': 'red',
    'neutral/ambiguous': 'gray'
}


# --- SECCIÓN 2: CARGA Y PROCESAMIENTO DE DATOS ---
print("Cargando y procesando los datos de anotaciones manuales...")

try:
    df = pd.read_csv("/home/jcuello/emotion_drift/data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv")
except FileNotFoundError:
    print("Error: No se encontró el archivo CSV. Asegúrate de que la ruta es correcta.")
    exit()

# Función de parseo robusta
def parse_labels_robust(label_string):
    if not isinstance(label_string, str) or label_string.strip() == "": return []
    cleaned_str = label_string.strip().strip('[]"\'')
    if not cleaned_str: return []
    labels = cleaned_str.split(',')
    return [label.strip().strip('\'"').lower() for label in labels if label.strip()]

# Aplicamos el parseo y los mapeos de sentimiento
df['ekman_labels_list'] = df['ekman_manual_label'].apply(parse_labels_robust)
df['go_emotions_labels_list'] = df['go_emotions_manual_label'].apply(parse_labels_robust)

df['prompt_sentiment'] = df['emotion_considered'].map(sentiment_map_prompt)

# Función para mapear una lista de emociones a una lista de sentimientos
def map_sentiments(label_list, sentiment_map):
    return [sentiment_map.get(label, 'unknown') for label in label_list]

df['ekman_sentiments'] = df['ekman_labels_list'].apply(map_sentiments, args=(sentiment_map_ekman,))
df['go_emotions_sentiments'] = df['go_emotions_labels_list'].apply(map_sentiments, args=(sentiment_map_go_emotions,))


# --- SECCIÓN 3: PREPARACIÓN DE DATOS PARA SANKEY ---

def prepare_sankey_links(df, response_sentiment_col):
    """
    Prepara los datos para el diagrama de Sankey, manejando múltiples etiquetas por respuesta.
    """
    # Seleccionamos las columnas relevantes y eliminamos filas sin sentimiento de prompt o respuesta
    links_df = df[['prompt_sentiment', response_sentiment_col]].dropna()
    
    # "Explotamos" la lista de sentimientos de la respuesta para crear un flujo por cada uno
    # Esto es clave para manejar las anotaciones con múltiples etiquetas
    links_df = links_df.explode(response_sentiment_col)
    
    # Contamos la frecuencia de cada flujo (Source -> Target)
    sankey_links = links_df.groupby(['prompt_sentiment', response_sentiment_col]).size().reset_index(name='value')
    sankey_links.rename(columns={'prompt_sentiment': 'source', response_sentiment_col: 'target'}, inplace=True)
    
    return sankey_links

# --- SECCIÓN 4: GENERACIÓN DE LOS DIAGRAMAS ---

def create_sankey_diagram(links_df, title):
    """
    Crea y muestra un diagrama de Sankey interactivo a partir de los datos de flujo.
    """
    # 1. Definir los nodos
    # Los nodos de origen necesitan un sufijo para distinguirlos visualmente de los de destino
    links_df['source'] = links_df['source'] + ' (Prompt)'
    
    all_nodes = list(pd.concat([links_df['source'], links_df['target']]).unique())
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    # 2. Mapear los nombres de los nodos a sus índices numéricos
    links_df['source_idx'] = links_df['source'].map(node_indices)
    links_df['target_idx'] = links_df['target'].map(node_indices)

    # 3. Asignar colores a los nodos
    node_colors = []
    for node in all_nodes:
        sentiment = node.replace(' (Prompt)', '')
        node_colors.append(color_map_sentiment.get(sentiment, 'blue'))

    # 4. Crear la figura
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=links_df['source_idx'],
            target=links_df['target_idx'],
            value=links_df['value']
        )
    )])

    fig.update_layout(title_text=title, font_size=12)
    fig.show()

# --- Ejecución para cada taxonomía ---

print("\nGenerando Diagrama de Sankey para la taxonomía Ekman...")
sankey_links_ekman = prepare_sankey_links(df, 'ekman_sentiments')
create_sankey_diagram(sankey_links_ekman, "Sentiment flow -\nEkman Taxonomy")

print("Generando Diagrama de Sankey para la taxonomía GoEmotions...")
sankey_links_go = prepare_sankey_links(df, 'go_emotions_sentiments')
create_sankey_diagram(sankey_links_go, "Sentiment flow -\nGoEmotions Taxonomy")

# %% Printing for error analysis

taxonomies = ["ekman", "go_emotions"]

df_list = []
for taxonomy in taxonomies:
    cols_of_interest = ["id","model",f"{taxonomy}_manual_label",f"{taxonomy}_labels",f"{taxonomy}_justification","response_text"]
    filter_column = f"broad_accuracy_{taxonomy}"

    df_filtered = df_merged[df_merged[filter_column]==0][cols_of_interest]
    df_list.append(df_filtered)

    print(f"=== {taxonomy.upper()} incorrect annotations (model hidden) ===")
    for row in df_filtered.iterrows():
        idx = row[0]
        data = row[1]

        response_id = data["id"]
        response_text = data["response_text"]
        manual_labels = data[f"{taxonomy}_manual_label"]
        model_labels = data[f"{taxonomy}_labels"]
        model_justification = data[f"{taxonomy}_justification"]
        print(f"\n\nData point {idx}")
        print(f"Response {response_id}: {response_text}")
        print(f"Manual labels: {manual_labels} \nModel labels: {model_labels} \nJustification: {model_justification}")
    
    print("="*50)

merged_errors = pd.merge(df_list[0], df_list[1], on=["id","response_text","model"], suffixes=("_ekman","_go_emotions"), how="outer")
merged_errors = merged_errors[merged_errors["model"] != "monologg/bert-base-cased"]
merged_errors_hidden_model = merged_errors[[col for col in merged_errors.columns if col != "model"]]

merged_errors.to_csv("/home/jcuello/emotion_drift/data/04_annotated/error_analysis_incorrect_annotations.csv")
merged_errors_hidden_model.to_csv("/home/jcuello/emotion_drift/data/04_annotated/error_analysis_incorrect_annotations_hidden_model.csv")
# %%
