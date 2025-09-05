#%%
import tiktoken
import google.generativeai as genai
import os

definitions_of_emotions = {"ekman_basic_emotions": f"""You must exclusively use the following taxonomy of emotions, paying attention to the given definitions of each emotional term:
                           *    Anger: The response to an interference with our pursuit of a goal we care about. Anger can also be triggered by someone attempting to harm us (physically or psychologically) or someone we care about. In addition to removing the obstacle or stopping the harm, anger often involves the wish to hurt the target.
                           *    Disgust: Arises as a feeling of repulsion or aversion towards something offensive. We can feel disgusted by something we perceive with our physical senses (sight, smell, touch, sound, taste), by the actions or appearances of people, and even by offensive ideas. Disgust contains a range of states with varying intensities from mild dislike to intense loathing.
                           *    Enjoyment: Typically arising from connection or sensory pleasure. We use the word enjoyment to describe a family of related pleasurable states, everything from peace to ecstasy.
                           *    Fear: Arises in response to the threat of harm, either physical, emotional, or psychological, real or imagined. Fear activates impulses to freeze or flee, serving an important role in keeping us safe as it mobilizes us to cope with potential danger.
                           *    Sadness: The response to the loss of an important object or a person to which you are very attached. Serves an important role in signaling a need to receive help or comfort. Sadness describes the range of emotional states from mild disappointment to extreme despair and anguish.
                           *    Surprise: Arises when we encounter sudden and unexpected events. As the briefest of the emotions, its function is to focus our attention on determining what is happening and whether or not it is dangerous. In the moment before we figure out what is occurring, before we switch to another emotion or no emotion, surprise itself can feel good or bad.
                           *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above.""",
                           
                           "go_emotions": f"""You must exclusively use the following taxonomy of emotions, paying attention to the given definitions of each emotional term:
                           *    Admiration: Finding something impressive or worthy of respect.
                           *    Amusement: Finding something funny or being entertained.
                           *    Anger: A strong feeling of displeasure or antagonism.
                           *    Annoyance: A feeling of mild anger or irritation.
                           *    Approval: Having or expressing a favorable opinion.
                           *    Caring: Displaying kindness and concern for others.
                           *    Confusion: Lack of understanding, uncertainty.
                           *    Curiosity: A strong desire to know or learn something.
                           *    Desire: A strong feeling of wanting something or wishing for something to happen.
                           *    Disappointment: Sadness or displeasure caused by the nonfulfillment of one’s hopes or expectations.
                           *    Disapproval: Having or expressing an unfavorable opinion.
                           *    Disgust: Revulsion or strong disapproval aroused by something unpleasant or offensive.
                           *    Embarrassment: Feeling of self-consciousness, shame, or awkwardness.
                           *    Excitement: Feeling of great enthusiasm and eagerness.
                           *    Fear: Being afraid or worried.
                           *    Gratitude: A feeling of thankfulness and appreciation.
                           *    Grief: Intense sorrow, especially caused by someone’s death.
                           *    Joy: A feeling of pleasure and happiness.
                           *    Love: A strong positive emotion of regard and affection.
                           *    Nervousness: A state of apprehension, worry and/or anxiety.
                           *    Optimism: Hopefulness and confidence about the future or the success of something.
                           *    Pride: Pleasure or satisfaction due to ones own achievements or the achievements of those with whom one is closely associated.
                           *    Realization: Feeling of becoming aware of something.
                           *    Relief: Reassurance and relaxation following release from anxiety or distress.
                           *    Remorse: Regret or guilty feeling.
                           *    Sadness: Emotional pain, sorrow.
                           *    Surprise: Feeling astonished, startled by something unexpected.
                           *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above.""",

                           "plutchik_wheel_of_emotions": f"""You must exclusively use the following taxonomy of emotions, paying attention to the given definitions of each emotional term:
                           *    Fear: Its evolutionary function is protection. It is triggered by the perception of a threat or imminent danger, which the mind interprets as a “dangerous” situation. This emotion triggers a response of flight or avoidance, with the ultimate goal of preserving physical integrity. Its intensity ranges from mild apprehension to paralyzing terror.
                           *    Anger: Has the adaptive function of destroying an obstacle that prevents the achievement of a goal. It arises when an individual is confronted with what they perceive as an “Enemy” or a barrier, prompting them to respond with aggressive behaviors to remove that barrier. This emotional state can manifest itself in a range from annoyance to rage.
                           *    Joy: Its evolutionary function focuses on reproduction and affiliation. It motivates to seek and retain valuable resources, such as a potential partner, food, or a significant achievement, functioning as a signal of success or gain. It is associated with a cognition of “Possession” and promotes behaviors such as courtship or celebration. It ranges from calm to ecstasy.
                           *    Sadness: Has evolved to promote social reintegration. It is triggered by the loss of a person or a valuable resource, generated by a cognition of “isolation.” It functions as a distress, manifesting itself through behaviors such as crying, which seek to attract support and comfort to facilitate recovery and return to the community. Its intensity ranges from a pensive state to deep grief.
                           *    Trust: Its function is affiliation and the creation of strong social bonds. It emerges when interacting with someone who is perceived as a “friend” or ally. This emotion is the basis for cooperation and mutual support within a group, fostering caring and collaborative behaviors. Its spectrum ranges from acceptance to admiration.
                           *    Disgust: Has the adaptive function of provoking rejection. It acts as a defense mechanism against poisoning or disease, activating in response to a repulsive object that is interpreted as “poison”. The behavioral response is visceral and immediate, including removal from the object to avoid contact. Its range extends from boredom to loathing or total aversion.
                           *    Anticipation: Is oriented toward the future. Its evolutionary function is exploration and preparation. It drives to investigate new possibilities or territories, starting from the cognitive question “What's out there?” It encourages behaviors such as planning, examining, and mapping, allowing the individual to prepare to find resources or opportunities. It ranges from an interest to a state vigilance.
                           *    Surprise: Its main adaptive function is orientation. It is activated by a novel and unexpected stimulus, prompting the immediate question “What is this?”. It causes an instant pause in the current action, forcing the individual to refocus their attention to quickly reassess the environment and decide how to respond to the new information. Its intensity can range from distraction to amazement.
                           *    Neutral: The absence of a predominant or clear emotion. The text is purely informational, factual, or does not express any discernible emotional state according to the categories above."""
                           }

#!! SACÁ ESTO DE ACÁ
gemini_api_key = ""
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

openai_encoding = tiktoken.get_encoding("o200k_base")

for taxonomy in definitions_of_emotions.keys():
    text = f"""**Role:** You are an expert data analyst specializing in Natural Language Processing (NLP), with a focus on emotion annotation. Your task is to analyze responses generated by an artificial intelligence to identify the emotions they express.

        **Objective:** Evaluate the following text objectively to identify the predominant emotion or emotions it contains. Your analysis must be rigorous, unbiased, and based solely on the evidence within the text.

        1. **Definitions of Emotion Categories:**
        {definitions_of_emotions[taxonomy]}
        
        2. **Annotation Rules:**
        You must follow these rules strictly:
        *   **Evidence Based**: Base your annotation exclusively on the content of the response. Do not infer emotions that are not supported by it.

        *   **Handling Mixed Emotions**:
        *   *   If a single emotion adequately describes the text's tone, use only that label.
        *   *   If two or more emotions are present, create a ranked list ordered by predominance. The first emotion in the list should be the strongest or most evident, followed by secondary emotions in descending order of importance.
        *   *   Your focus in these cases is to determine the overall 'weight' or 'proportion' of each emotion in the text as a whole. **Do not** list emotions in the order they appear. A brief emotion at the beginning is less important than a sustained emotion that permeates the rest of the text.
        *   *   If a text begins with an expression of "emotion 'A'", but the remaining 90% of the content elaborates on a topic with a clear tone of "emotion 'B'", then "emotion 'B'" is the most predominant emotion, and "emotion 'A'" is secondary. Then, **do not** list emotions in the order they appear. The output in this situation should be: ["emotion 'b'", "emotion 'a'"]

        *   **Concise Justification**: Provide a brief justification (1-2 sentences) that explains your choice, quoting or paraphrasing key phrases from the text that support your analysis.

        3. **Required Output Format:**
        Your response must be a single JSON code block with the following structure:

        ```json
        {{
        "emotions": ["Emotion_1", "Emotion_2", ...],
        "justification": "Your explanation here."
        }}```

        Your entire response must be ONLY in single JSON code block. Do not include any introductory or concluding sentences outside the code block.

        4. **Text to Annotate:**
        Here is the text to analyze:"""

    tokens = openai_encoding.encode(text)
    num_tokens_openai = len(tokens)

    num_tokens_google = gemini_model.count_tokens(text)

    print("="*50)
    print(f"Rubric: '{taxonomy}'")
    print(f"Num of tokens for GPT-5: {num_tokens_openai}")
    print(f"Num of tokens for Gemini-2.5-pro: {num_tokens_google}")


# %%
