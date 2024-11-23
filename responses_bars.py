import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
random.seed(1)

dataset_path = os.path.join("data", "AACEmotionRepresentation.xlsx")

def plot_single_chart(series: pd.Series, title: str, xlabel: str, ylabel: str):
    choice_counts = series.astype(int)
    choice_counts.index = choice_counts.index.astype(int)
    choice_counts.sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_multiple_bars(serieses: [pd.Series], title: str, xlabel: str, ylabel: str):
    labels = serieses[0].index.tolist()

    if len(labels) == 4:
        labels = [{1: "Words", 2: "VAD", 3: "VAD_Numeric", 4: "Emojis"}.get(x) for x in labels]

    bar_width, x_pos = 0.2, np.arange(len(labels))

    plt.bar(x_pos - 1.5 * bar_width, serieses[0].values, width=bar_width, label='Words', color='blue')
    plt.bar(x_pos - 0.5 * bar_width, serieses[1].values, width=bar_width, label='VAD', color='green')
    plt.bar(x_pos + 0.5 * bar_width, serieses[2].values, width=bar_width, label='VAD Numeric', color='red')
    plt.bar(x_pos + 1.5 * bar_width, serieses[3].values, width=bar_width, label='Emojis', color='orange')

    plt.xlabel(xlabel)  # i.e. When people were told the emotion in words, they picked the sentence prompted with Words most often, with Emojis second-most often, etc.
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x_pos, labels)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.show()


def plot_choice_counts(data: pd.DataFrame, which_data: str):
    questions_responses = data
    choice_counts = pd.Series(questions_responses.values.flatten()).value_counts()
    print(f"For {which_data}, for each option there were were:", choice_counts,
          f"There were {questions_responses.count().sum()} total questions answered.", sep="\n")
    plot_single_chart(series=choice_counts, title=f"{which_data} Questions Responses", xlabel="Choice", ylabel="Count")
    return choice_counts.sort_index()


dataset = pd.read_excel(dataset_path)

# ========== #

words_data = dataset[dataset['representation_type'] == "1"]

words_questions_col_names = [f"Q19_{n}" for n in range(1, 37)]
words_choice_counts = plot_choice_counts(data=words_data[words_questions_col_names], which_data="Words")

words_sliders_convey_names = [f"{n}_Q110_1_1" for n in range(1, 37)]
words_sliders_convey_counts = plot_choice_counts(data=words_data[words_sliders_convey_names], which_data="Words; Convey")

words_sliders_idsay_names = [f"{n}_Q110_1_2" for n in range(1, 37)]
words_sliders_idsay_counts = plot_choice_counts(data=words_data[words_sliders_idsay_names], which_data="Words; I'd say")

words_sliders_smesay_names = [f"{n}_Q110_1_3" for n in range(1, 37)]
words_sliders_smesay_counts = plot_choice_counts(data=words_data[words_sliders_smesay_names], which_data="Words; Someone else'd say")

# ============================== #

vad_data = dataset[dataset['representation_type'] == "2"]

vad_questions_col_names = [f"QID{246+n}" for n in range(0, 36)]
vad_choice_counts = plot_choice_counts(data=vad_data[vad_questions_col_names], which_data="VAD")

vad_sliders_convey_names = [f"{n}_Q31_1" for n in range(1, 37)]
vad_sliders_convey_counts = plot_choice_counts(data=vad_data[vad_sliders_convey_names], which_data="VAD; Convey")

vad_sliders_idsay_names = [f"{n}_Q31_2" for n in range(1, 37)]
vad_sliders_idsay_counts = plot_choice_counts(data=vad_data[vad_sliders_idsay_names], which_data="VAD; I'd say")

vad_sliders_smesay_names = [f"{n}_Q31_3" for n in range(1, 37)]
vad_sliders_smesay_counts = plot_choice_counts(data=vad_data[vad_sliders_smesay_names], which_data="VAD; Someone else'd say")

# ============================== #

vadnum_data = dataset[dataset['representation_type'] == "3"]
vadnum_questions_col_names = [f"QID{510+n}" for n in range(0, 36)]
vadnum_questions_responses = vadnum_data[vadnum_questions_col_names]

# Sampling a random set of columns per row. Seed is set at the top of the file.
cols_rand = list(vadnum_questions_responses.columns[1:])
vadnum_questions_responses[cols_rand] = vadnum_questions_responses.apply(
    lambda row: [row[col] if col in random.sample(cols_rand, 10) else np.nan for col in cols_rand],
    axis=1, result_type='expand')

vadnum_questions_col_names = [f"QID{246+n}" for n in range(0, 36)]
vadnum_choice_counts = plot_choice_counts(data=vadnum_questions_responses, which_data="VAD Numeric")

vadnum_sliders_convey_names = [f"{n}_Q402_1" for n in range(1, 37)]
vadnum_sliders_convey_counts = plot_choice_counts(data=vadnum_data[vadnum_sliders_convey_names], which_data="VAD Numeric; Convey")

vadnum_sliders_idsay_names = [f"{n}_Q402_2" for n in range(1, 37)]
vadnum_sliders_idsay_counts = plot_choice_counts(data=vadnum_data[vadnum_sliders_idsay_names], which_data="VAD Numeric; I'd say")

vadnum_sliders_smesay_names = [f"{n}_Q402_2" for n in range(1, 37)]
vadnum_sliders_smesay_counts = plot_choice_counts(data=vadnum_data[vadnum_sliders_smesay_names], which_data="VAD Numeric; Someone else'd say")

# ============================== #

emojis_data = dataset[dataset['representation_type'] == "4"]

emojis_questions_col_names = [f"QID{578+n}" for n in range(0, 36)]
emojis_choice_counts = plot_choice_counts(data=emojis_data[emojis_questions_col_names], which_data="Emojis")

emojis_sliders_convey_names = [f"{n}_Q403_1" for n in range(1, 37)]
emojis_sliders_convey_counts = plot_choice_counts(data=emojis_data[emojis_sliders_convey_names], which_data="Emojis; Convey")

emojis_sliders_idsay_names = [f"{n}_Q403_2" for n in range(1, 37)]
emojis_sliders_idsay_counts = plot_choice_counts(data=emojis_data[emojis_sliders_idsay_names], which_data="Emojis; I'd say")

emojis_sliders_smesay_names = [f"{n}_Q403_3" for n in range(1, 37)]
emojis_sliders_smesay_counts = plot_choice_counts(data=emojis_data[emojis_sliders_smesay_names], which_data="Emojis; Someone else'd say")

# ============================== #

plot_multiple_bars(serieses=[words_choice_counts, vad_choice_counts, vadnum_choice_counts, emojis_choice_counts],
                   title="Select the Best Sentence", xlabel="LLM-Generated Sentence Picked", ylabel="Number of Times")

plot_multiple_bars(serieses=[words_sliders_convey_counts, vad_sliders_convey_counts, vadnum_sliders_convey_counts, emojis_sliders_convey_counts],
                   title="How well does this sentence convey the emotion?", xlabel="Score for the given sentence given the emotion medium", ylabel="Number of Times")

plot_multiple_bars(serieses=[words_sliders_idsay_counts, vad_sliders_idsay_counts, vadnum_sliders_idsay_counts, emojis_sliders_idsay_counts],
                   title="How much does this sounds like something I'd say?", xlabel="Score for the given sentence given the emotion medium", ylabel="Number of Times")

plot_multiple_bars(serieses=[words_sliders_smesay_counts, vad_sliders_smesay_counts, vadnum_sliders_smesay_counts, emojis_sliders_smesay_counts],
                   title="How much does this sounds like something someone else'd say?", xlabel="Score for the given sentence given the emotion medium", ylabel="Number of Times")



