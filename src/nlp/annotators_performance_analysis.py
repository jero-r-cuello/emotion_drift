import os
import re
from collections import Counter
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import statsmodels.stats.proportion as smp
from tqdm import tqdm


# Model annotations to be evaluated. 'models_annotation_final' contains frontier reasoning models, while 'gpt-5-mini-consolidated_annotations' is a more focused set of GPT-5 mini annotations.
MODEL_ANNOTATIONS_PATH = "data/04_annotated/gpt-5-mini-consolidated_annotations.csv" # "data/04_annotated/models_annotations_final.csv" #
# Ground truth to be compared against
MANUAL_ANNOTATIONS_PATH = "data/04_annotated/anotacion_manual_generated_responses - Sheet1.csv"
OUTPUT_ERRORS_PATH = "data/04_annotated/error_analysis_incorrect_annotations.csv"
OUTPUT_ERRORS_HIDDEN_PATH = "data/04_annotated/error_analysis_incorrect_annotations_hidden_model.csv"
FIGS_PATH = f'figures/annottors_performance_analysis/{MODEL_ANNOTATIONS_PATH.split("/")[-1].replace(".csv", "")}'
os.makedirs(FIGS_PATH, exist_ok=True)

# Taxonomies
DEFINITIONS_OF_EMOTIONS = {"ekman_basic_emotions": f"""You must exclusively use the following taxonomy of emotions:
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

SENTIMENT_MAP_PROMPT = {
    "agony": "negative", "anger": "negative", "delight": "positive", 
    "disgust": "negative", "fear": "negative", "hope": "positive", 
    "joy": "positive", "love": "positive", "sadness": "negative", 
    "shame": "negative", "surprise": "neutral/ambiguous"
}
SENTIMENT_MAP_EKMAN = {
    "disgust": "negative", "anger": "negative", "fear": "negative", 
    "sadness": "negative", "enjoyment": "positive", 
    "surprise": "neutral/ambiguous", "neutral": "neutral/ambiguous"
}
SENTIMENT_MAP_GO_EMOTIONS = {
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
    "neutral": "neutral/ambiguous" 
}
COLOR_MAP_SENTIMENT = {
    "positive": "green",
    "negative": "red",
    "neutral/ambiguous": "gray"
}

# Kappa Settings
N_BOOTSTRAPS = 2000
CONFIDENCE_LEVEL = 0.95
WEIGHTS = {"ekman": 0.5, "go_emotions": 0.5}
RANDOM_SEED = 42


def extract_labels_from_definition(text):
    labels = re.findall(r"\*\s*([^:]+):", text)
    return [label.strip().lower() for label in labels]

EKMAN_LABELS_DEFINED = extract_labels_from_definition(DEFINITIONS_OF_EMOTIONS["ekman_basic_emotions"])
GO_EMOTIONS_LABELS_DEFINED = extract_labels_from_definition(DEFINITIONS_OF_EMOTIONS["go_emotions"])

def parse_labels_robust(label_string):
    if not isinstance(label_string, str) or label_string.strip() == "": return []
    cleaned_str = label_string.strip().strip('[]"\'')
    if not cleaned_str: return []
    return [label.strip().strip('\'"').lower() for label in cleaned_str.split(",") if label.strip()]

def get_invalid_labels(label_list, valid_labels_set):
    if not label_list: return []
    return [label for label in label_list if label not in valid_labels_set]

def map_sentiments(label_list, sentiment_map):
    return [sentiment_map.get(label, "unknown") for label in label_list]


def load_and_preprocess_data():
    print("Loading and processing data...")
    try:
        manual = pd.read_csv(MANUAL_ANNOTATIONS_PATH)
        models = pd.read_csv(MODEL_ANNOTATIONS_PATH)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        exit(1)

    if "model" in models.columns:
        models.rename(columns={"model": "annotator"}, inplace=True)
    
    df = pd.merge(manual, models, on="response_text")

    # Parse all label lists once
    label_mappings = {
        "ekman_manual_labels_list": "ekman_manual_label",
        "go_emotions_manual_labels_list": "go_emotions_manual_label",
        "ekman_annotator_labels_list": "ekman_labels",
        "go_emotions_annotator_labels_list": "go_emotions_labels"
    }
    for new_col, old_col in label_mappings.items():
        df[new_col] = df[old_col].apply(parse_labels_robust)

    # Calculate Sentiments for Sankey
    df["prompt_sentiment"] = df["emotion_considered"].map(SENTIMENT_MAP_PROMPT)
    df["ekman_sentiments"] = df["ekman_manual_labels_list"].apply(map_sentiments, args=(SENTIMENT_MAP_EKMAN,))
    df["go_emotions_sentiments"] = df["go_emotions_manual_labels_list"].apply(map_sentiments, args=(SENTIMENT_MAP_GO_EMOTIONS,))

    return df

# Plots
def plot_label_frequencies(df, taxonomy_name, col_prefix, defined_labels):
    print(f"Generating frequency plots for {taxonomy_name}...")
    manual_col = f'{col_prefix}_manual_labels_list'
    annotator_col = f'{col_prefix}_annotator_labels_list'

    gt_unique = df.drop_duplicates(subset=["response_text"])[["response_text", manual_col]].copy()
    gt_unique["source"] = "Ground Truth"
    gt_labels = gt_unique[["source", manual_col]].explode(manual_col).dropna()
    gt_labels.rename(columns={manual_col: "label"}, inplace=True)

    model_labels = df[["annotator", annotator_col]].explode(annotator_col).dropna()
    model_labels.rename(columns={"annotator": "source", annotator_col: "label"}, inplace=True)

    combined_labels = pd.concat([gt_labels[["source", "label"]], model_labels[["source", "label"]]], ignore_index=True)
    combined_counts = combined_labels.groupby(["source", "label"]).size().reset_index(name="count")

    sources = ["Ground Truth"] + sorted(df["annotator"].unique())
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows=len(sources), ncols=1, figsize=(14, 6 * len(sources)), sharex=True)
    if len(sources) == 1: axes = [axes]
    fig.suptitle(f'Freq. of labels from {taxonomy_name.title()}', fontsize=20, y=0.995)

    for i, source in enumerate(sources):
        ax = axes[i]
        df_plot = combined_counts[combined_counts["source"] == source]
        
        sns.barplot(data=df_plot, x="label", y="count", hue="label", palette="viridis", 
                    order=defined_labels, legend=False, ax=ax)
        
        ax.set_title(source, fontsize=15, loc="left", pad=10)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=90)
        
        for p in ax.patches:
            # Check if height is valid and greater than 0
            if pd.notna(p.get_height()) and p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha="center", va="center", xytext=(0, 9), textcoords="offset points")

    axes[-1].set_xlabel("Emotion label", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'{FIGS_PATH}/{taxonomy_name}_label_frequencies.png', dpi=300)
    plt.close()
    

def plot_number_of_labels(df, taxonomy_name, col_prefix):
    print(f"Generating complexity graph for {taxonomy_name}...")
    manual_col = f'{col_prefix}_manual_labels_list'
    annotator_col = f'{col_prefix}_annotator_labels_list'

    gt_unique = df[["response_text", manual_col]].groupby("response_text").first().reset_index()
    gt_unique["num_labels"] = gt_unique[manual_col].str.len()
    gt_unique["source"] = "Ground Truth"

    df_models = df.copy()
    df_models["num_labels"] = df_models[annotator_col].str.len()
    df_models.rename(columns={"annotator": "source"}, inplace=True)
    
    combined_data = pd.concat([gt_unique[["source", "num_labels"]], df_models[["source", "num_labels"]]], ignore_index=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.countplot(data=combined_data, x="num_labels", hue="source", ax=ax)
    ax.set_title(f'Amount of labels per response from ({taxonomy_name.title()})', fontsize=18, pad=20)
    ax.set_xlabel("Amount of labels", fontsize=14)
    ax.set_ylabel("Count of responses", fontsize=14)
    ax.legend(title="Annotator")
    plt.tight_layout()
    plt.savefig(f'{FIGS_PATH}/{taxonomy_name}_num_labels_distribution.png', dpi=300)
    

def plot_cooccurrence_heatmap(df, taxonomy_name, col_name):
    print(f"Generating co-occurrence heatmap for {taxonomy_name}...")
    gt_unique = df[["response_text", col_name]].groupby("response_text").first().reset_index()
    label_lists = gt_unique[gt_unique[col_name].str.len() > 1][col_name]
    
    if label_lists.empty:
        print(f"There are no multiple tags in Ground Truth {taxonomy_name}. It is omitted.")
        return
        
    co_occurrences = Counter(pair for labels in label_lists for pair in combinations(sorted(labels), 2))
    all_labels = sorted(list(set(label for sublist in gt_unique[col_name] for label in sublist)))
    co_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)
    
    for (label1, label2), count in co_occurrences.items():
        co_matrix.loc[label1, label2] = count
        co_matrix.loc[label2, label1] = count

    plt.figure(figsize=(12, 10))
    sns.heatmap(co_matrix, cmap="viridis", annot=True, fmt="d", linewidths=.5)
    plt.title(f'Co-occurrence of Tags in Ground Truth ({taxonomy_name.title()})', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'{FIGS_PATH}/{taxonomy_name}_coocurrence_heatmap.png', dpi=300)
    

def analyze_hallucinations(df):
    print("\nAnalyzing label hallucinations...")
    model_to_exclude = "monologg/bert-base-cased"
    df_analysis = df[df["annotator"] != model_to_exclude].copy()
    
    ekman_set = set(EKMAN_LABELS_DEFINED)
    go_set = set(GO_EMOTIONS_LABELS_DEFINED)

    df_analysis["ekman_halluc"] = df_analysis["ekman_annotator_labels_list"].apply(get_invalid_labels, args=(ekman_set,))
    df_analysis["go_halluc"] = df_analysis["go_emotions_annotator_labels_list"].apply(get_invalid_labels, args=(go_set,))

    data_plot = []
    print("="*50 + "\nHallucinated labels by annotator\n" + "="*50)
    for annotator, group in df_analysis.groupby("annotator"):
        ek_halluc = [w for sub in group["ekman_halluc"] for w in sub]
        go_halluc = [w for sub in group["go_halluc"] for w in sub]
        
        data_plot.append({"annotator": annotator, "ekman_hallucinations": len(ek_halluc), "go_emotions_hallucinations": len(go_halluc)})
        
        if ek_halluc: print(f"- Annotator: {annotator} (Ekman)\n  Words: {sorted(list(set(ek_halluc)))}")
        if go_halluc: print(f"- Annotator: {annotator} (GoEmotions)\n  Words: {sorted(list(set(go_halluc)))}")

    plot_df = pd.melt(pd.DataFrame(data_plot), id_vars="annotator", value_vars=["ekman_hallucinations", "go_emotions_hallucinations"], var_name="Taxonomy", value_name="Count")
    plot_df["Taxonomy"] = plot_df["Taxonomy"].str.replace("_hallucinations", "").str.title()

    if plot_df["Count"].sum() > 0:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=plot_df, x="annotator", y="Count", hue="Taxonomy", ax=ax)
        ax.set_title('Count of "hallucinated" labels (out of taxonomy)', fontsize=16, pad=20)
        plt.xticks(rotation=30, ha="right")
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha="center", va="center", xytext=(0, 9), textcoords="offset points")
        plt.tight_layout()
        plt.savefig(f'{FIGS_PATH}/hallucinated_labels_count.png', dpi=300)
        
    else:
        print("No hallucinated labels were found.")

def plot_confusion_matrices(df, taxonomy_name, gt_col, pred_col, all_labels, figsize=(18, 15), threshold=0.001, errors_only=False):
    print(f"\nGenerating confusion matrices: {taxonomy_name.title()}")
    for annotator in df["annotator"].unique():
        df_model = df[df["annotator"] == annotator].copy()
        y_true, y_pred = df_model[gt_col].str[0], df_model[pred_col].str[0]
        df_plot = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()

        if df_plot.empty: continue

        cm = confusion_matrix(df_plot["y_true"], df_plot["y_pred"], labels=all_labels, normalize="true")
        if errors_only: np.fill_diagonal(cm, 0)

        fig, ax = plt.subplots(figsize=figsize)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels).plot(ax=ax, cmap="Blues", xticks_rotation="vertical", include_values=False)
        
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                val = cm[i, j]
                if val > threshold:
                    ax.text(j, i, f'{val:.2f}', ha="center", va="center", color="white" if val > 0.5 else "black")

        suffix = " (Error map)" if errors_only else ""
        ax.set_title(f'Normalized CM - {annotator}\n{taxonomy_name.title()}{suffix}', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{FIGS_PATH}/{taxonomy_name}_confusion_matrix_{annotator.split("/")[1] if "/" in annotator else annotator}{"_errors" if errors_only else ""}.png', dpi=300)
        

def calculate_accuracies_and_plot(df):
    print("\nCalculating accuracies and confidence intervals...")
    def acc(r, gt, p):
        try: return 1 if r[gt].strip("[]").split(",")[0].strip().lower() == re.findall(r"\'(.*?)\'", r[p])[0].strip().lower() else 0
        except: return "error"
    
    # For broad accuracy, we check if the model's top prediction is among any of the human labels for that response
    def broad_acc(r, gt, p):
        try:
            mod = re.findall(r"\'(.*?)\'", r[p])[0].strip().lower()
            man = [l.strip().lower() for l in r[gt].strip("[]").split(",")]
            return 1 if mod in man else 0
        except: return "error"

    df["acc_ekman"] = df.apply(lambda r: acc(r, "ekman_manual_label", "ekman_labels"), axis=1)
    df["acc_go"] = df.apply(lambda r: acc(r, "go_emotions_manual_label", "go_emotions_labels"), axis=1)
    df["broad_ekman"] = df.apply(lambda r: broad_acc(r, "ekman_manual_label", "ekman_labels"), axis=1)
    df["broad_go"] = df.apply(lambda r: broad_acc(r, "go_emotions_manual_label", "go_emotions_labels"), axis=1)

    results = []
    for model, group in df.groupby("annotator"):
        res = {"model": model}
        for col in ["acc_ekman", "broad_ekman", "acc_go", "broad_go"]:
            valid = pd.to_numeric(group[col], errors="coerce").dropna()
            count, nobs = valid.sum(), len(valid)
            mean = count / nobs if nobs > 0 else 0
            low, upp = smp.proportion_confint(count, nobs, alpha=0.05, method="beta") if nobs > 0 else (0, 0)
            res.update({f'{col}_mean': mean, f'{col}_low': low, f'{col}_upp': upp})
        results.append(res)
        
    plot_data = pd.DataFrame(results).set_index("model")
    print(plot_data.filter(like="_mean").round(4) * 100)

    plot_data["gain_ekman"] = plot_data["broad_ekman_mean"] - plot_data["acc_ekman_mean"]
    plot_data["gain_go"] = plot_data["broad_go_mean"] - plot_data["acc_go_mean"]

    x = np.arange(len(plot_data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(x - width/2, plot_data["acc_ekman_mean"], width, label="Ekman Acc", color="cornflowerblue")
    ax.bar(x - width/2, plot_data["gain_ekman"], width, bottom=plot_data["acc_ekman_mean"], label="Ekman Broad", color="deepskyblue", 
           yerr=[plot_data["broad_ekman_mean"]-plot_data["broad_ekman_low"], plot_data["broad_ekman_upp"]-plot_data["broad_ekman_mean"]], capsize=4)
    
    ax.bar(x + width/2, plot_data["acc_go_mean"], width, label="GoEmotions Acc", color="darkred")
    ax.bar(x + width/2, plot_data["gain_go"], width, bottom=plot_data["acc_go_mean"], label="GoEmotions Broad", color="tomato",
           yerr=[plot_data["broad_go_mean"]-plot_data["broad_go_low"], plot_data["broad_go_upp"]-plot_data["broad_go_mean"]], capsize=4)

    ax.set_title("Accuracy per model and taxonomy", fontsize=18)
    ax.set_xticks(x); ax.set_xticklabels(plot_data.index, rotation=30, ha="right")
    ax.set_yticklabels([f'{int(tick*100)}%' for tick in ax.get_yticks()])
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGS_PATH}/accuracy_comparison.png', dpi=300)
    
    return df

def calculate_kappas_and_plot(df):
    print("\nCalculating Broad Kappas and Bootstrapping...")
    all_gt_ekman = [label for sub in df["ekman_manual_labels_list"] for label in sub]
    all_gt_go = [label for sub in df["go_emotions_manual_labels_list"] for label in sub]

    def calc_kappa(df_sub, gt_col, pred_col, gt_universe):
        preds = df_sub[pred_col].str[0]
        eff_gt = df_sub.apply(lambda r: r[pred_col][0] if r[pred_col][0] in r[gt_col] else r[gt_col][0], axis=1)
        all_l = list(set(gt_universe) | set(preds.unique()))
        return cohen_kappa_score(eff_gt, preds, labels=all_l)

    results = []
    for annotator in tqdm(df["annotator"].unique(), desc="Kappas por modelo"):
        df_sub = df[df["annotator"] == annotator].copy()
        mask = (df_sub["ekman_manual_labels_list"].str.len() > 0) & (df_sub["go_emotions_manual_labels_list"].str.len() > 0) & \
               (df_sub["ekman_annotator_labels_list"].str.len() > 0) & (df_sub["go_emotions_annotator_labels_list"].str.len() > 0)
        df_clean = df_sub[mask]
        
        if len(df_clean) == 0: continue
        
        k_ek = calc_kappa(df_clean, "ekman_manual_labels_list", "ekman_annotator_labels_list", all_gt_ekman)
        k_go = calc_kappa(df_clean, "go_emotions_manual_labels_list", "go_emotions_annotator_labels_list", all_gt_go)
        k_w = (WEIGHTS["ekman"] * k_ek) + (WEIGHTS["go_emotions"] * k_go)

        boot_w, boot_ek, boot_go = [], [], []
        for i in range(N_BOOTSTRAPS):
            d_b = df_clean.sample(n=len(df_clean), replace=True, random_state=RANDOM_SEED + i)
            b_ek = calc_kappa(d_b, "ekman_manual_labels_list", "ekman_annotator_labels_list", all_gt_ekman)
            b_go = calc_kappa(d_b, "go_emotions_manual_labels_list", "go_emotions_annotator_labels_list", all_gt_go)
            boot_ek.append(b_ek); boot_go.append(b_go); boot_w.append((WEIGHTS["ekman"]*b_ek) + (WEIGHTS["go_emotions"]*b_go))
        
        alpha = (1 - CONFIDENCE_LEVEL) / 2
        results.append({
            "Annotator": annotator, "W_Kappa": k_w, 
            "W_Low": np.percentile(boot_w, alpha*100), "W_Upp": np.percentile(boot_w, (1-alpha)*100),
            "Ek_Kappa": k_ek, "Ek_Low": np.percentile(boot_ek, alpha*100), "Ek_Upp": np.percentile(boot_ek, (1-alpha)*100),
            "Go_Kappa": k_go, "Go_Low": np.percentile(boot_go, alpha*100), "Go_Upp": np.percentile(boot_go, (1-alpha)*100)
        })

    plot_df = pd.DataFrame(results).sort_values("W_Kappa", ascending=False)
    print("\nKappa results:\n", plot_df[["Annotator", "W_Kappa", "Ek_Kappa", "Go_Kappa"]].round(3))

    x = np.arange(len(plot_df)); width = 0.25
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(x - width/2, plot_df["Ek_Kappa"], width, label="Ekman", color="cornflowerblue", yerr=[plot_df["Ek_Kappa"]-plot_df["Ek_Low"], plot_df["Ek_Upp"]-plot_df["Ek_Kappa"]], capsize=4)
    ax.bar(x, plot_df["W_Kappa"], width, label="Weighted Avg", color="mediumseagreen", yerr=[plot_df["W_Kappa"]-plot_df["W_Low"], plot_df["W_Upp"]-plot_df["W_Kappa"]], capsize=4)
    ax.bar(x + width/2, plot_df["Go_Kappa"], width, label="GoEmotions", color="salmon", yerr=[plot_df["Go_Kappa"]-plot_df["Go_Low"], plot_df["Go_Upp"]-plot_df["Go_Kappa"]], capsize=4)
    
    ax.set_title("Kappa score per annotator and taxonomy", fontsize=18)
    ax.set_xticks(x); ax.set_xticklabels(plot_df["Annotator"], rotation=30, ha="right")
    ax.set_ylim(bottom=0, top=1); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGS_PATH}/kappa_comparison.png', dpi=300)
    

def plot_sankey(df, target_col, title):
    print(f"\nGenerating Sankey figure for {title}...")
    links = df[["prompt_sentiment", target_col]].dropna().explode(target_col)
    links = links.groupby(["prompt_sentiment", target_col]).size().reset_index(name="value")
    links.columns = ["source", "target", "value"]
    links["source"] = links["source"] + " (Prompt)"
    
    nodes = list(pd.concat([links["source"], links["target"]]).unique())
    idx_map = {n: i for i, n in enumerate(nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, label=nodes, color=[COLOR_MAP_SENTIMENT.get(n.replace(" (Prompt)", ""), "blue") for n in nodes]),
        link=dict(source=links["source"].map(idx_map), target=links["target"].map(idx_map), value=links["value"])
    )])
    fig.update_layout(title_text=title, font_size=12)
    plt.savefig(f'{FIGS_PATH}/{target_col}_sankey.png', dpi=300)

def run_error_analysis(df):
    print("\nRunning error analysis...")
    dfs = []
    for tax, broad_col in [("ekman", "broad_ekman"), ("go_emotions", "broad_go")]:
        cols = ["id", "annotator", f"{tax}_manual_label", f"{tax}_labels", f"{tax}_justification", "response_text"]
        df_fail = df[df[broad_col] == 0][cols].rename(columns={"annotator": "model"})
        dfs.append(df_fail)
        
        print(f"\n=== {tax.upper()} Errors ===")
        for _, r in df_fail.head(3).iterrows(): # Show only top 3 on terminal to avoid clutter
            print(f"ID {r['id']}: {r['response_text'][:50]}... | Manual: {r[f'{tax}_manual_label']} | Model: {r[f'{tax}_labels']}")
    
    merged = pd.merge(dfs[0], dfs[1], on=["id", "response_text", "model"], suffixes=("_ekman", "_go_emotions"), how="outer")
    merged = merged[merged["model"] != "monologg/bert-base-cased"]
    
    merged.to_csv(OUTPUT_ERRORS_PATH, index=False)
    merged.drop(columns=["model"]).to_csv(OUTPUT_ERRORS_HIDDEN_PATH, index=False)
    print(f"\nReports saved to:\n - {OUTPUT_ERRORS_PATH}\n - {OUTPUT_ERRORS_HIDDEN_PATH}")


if __name__ == "__main__":
    print("Starting analysis of annotations...")
    
    # 1. Load Data
    df_main = load_and_preprocess_data()
    print(f"Data successfully combined. Total rows: {len(df_main)}")

    # 2. Frequencies
    plot_label_frequencies(df_main, "Ekman", "ekman", EKMAN_LABELS_DEFINED)
    plot_label_frequencies(df_main, "GoEmotions", "go_emotions", GO_EMOTIONS_LABELS_DEFINED)

    # 3. Complexities
    plot_number_of_labels(df_main, "Ekman", "ekman")
    plot_number_of_labels(df_main, "GoEmotions", "go_emotions")

    # 4. Co-occurrences
    plot_cooccurrence_heatmap(df_main, "Ekman", "ekman_manual_labels_list")
    plot_cooccurrence_heatmap(df_main, "GoEmotions", "go_emotions_manual_labels_list")

    # 5. Hallucinations
    analyze_hallucinations(df_main)

    # 6. Confusion Matrices
    plot_confusion_matrices(df_main, "Ekman", "ekman_manual_labels_list", "ekman_annotator_labels_list", EKMAN_LABELS_DEFINED, threshold=0.001)
    plot_confusion_matrices(df_main, "GoEmotions", "go_emotions_manual_labels_list", "go_emotions_annotator_labels_list", GO_EMOTIONS_LABELS_DEFINED, threshold=0.001, errors_only=True)

    # 7. Accuracies
    df_main = calculate_accuracies_and_plot(df_main)

    # 8. Kappa score
    calculate_kappas_and_plot(df_main)

    # 9. Sankey Diagrams
    plot_sankey(df_main, "ekman_sentiments", "Sentiment flow - Ekman Taxonomy")
    plot_sankey(df_main, "go_emotions_sentiments", "Sentiment flow - GoEmotions Taxonomy")

    # 10. Error analysis export (for qualitative review)
    run_error_analysis(df_main)

    print("\nPipeline completed.")