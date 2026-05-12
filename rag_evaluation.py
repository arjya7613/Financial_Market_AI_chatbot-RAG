# =========================================================
# IMPORTS
# =========================================================

import pandas as pd

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag_pipeline1 import run_financial_rag


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# GROUND TRUTH
ground_truth = [

    {
        "question":
        "What was Apple's total net sales in 2025?",

        "reference":
        "Apple reported total net sales of $416,161 million in 2025."
    },

    {
        "question":
        "Which region had declining revenue in 2025?",

        "reference":
        "Greater China experienced declining revenue in 2025."
    },

    {
        "question":
        "What was Apple's Services segment growth in 2025?",

        "reference":
        "Apple's Services segment grew 14% in 2025."
    },

    {
        "question":
        "How many employees did Apple have in 2025?",

        "reference":
        "Apple had around 166,000 employees in 2025."
    },

    {
        "question":
        "What dividend did Apple declare in May 2025?",

        "reference":
        "Apple declared a dividend of $0.26 per share in May 2025."
    }

]

# ROUGE + BLEU
scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)
smoothie = SmoothingFunction().method1

# RESULTS
results = []

print("\n" + "=" * 60)
print("RUNNING RAG EVALUATION")
print("=" * 60)

# MAIN LOOP
for item in ground_truth:
    question = item["question"]
    reference = item["reference"]
    print(f"\nQuestion: {question}")
    # RUN RAG PIPELINE
    response = run_financial_rag(
        query=question,
        mode="simple",
        evaluation=True
    )
    generated = response["answer"]
    latency = response["latency"]

    # BLEU SCORE
    bleu = sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=smoothie
    )

    # ROUGE SCORE
    rouge = scorer.score(
        reference,
        generated
    )
    rouge_l = rouge["rougeL"].fmeasure

    # RELEVANCE SCORE
    ref_embedding = embedding_model.embed_query(
        reference
    )
    gen_embedding = embedding_model.embed_query(
        generated
    )
    relevance = cosine_similarity(
        [ref_embedding],
        [gen_embedding]
    )[0][0]

    # STORE RESULTS
    results.append({
        "Question": question,
        "BLEU": round(bleu, 4),
        "ROUGE-L": round(rouge_l, 4),
        "Relevance": round(
            float(relevance),
            4
        ),
        "Latency": latency
    })

# CREATE DATAFRAME
df = pd.DataFrame(results)

# SHOW RESULTS
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(df)

# AVERAGE SCORES
print("\n" + "=" * 60)
print("AVERAGE SCORES")
print("=" * 60)

print(df.mean(numeric_only=True))

# SAVE RESULTS
df.to_csv(
    "rag_evaluation_results.csv",
    index=False
)
print("\nResults saved to rag_evaluation_results.csv")