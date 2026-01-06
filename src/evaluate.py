"""
Task 3: RAG Qualitative Evaluation

Runs 10 test questions through the RAG pipeline and outputs evaluation results.

Usage:
    python -m src.evaluate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGPipeline
from datetime import datetime

# 10 Representative test questions
TEST_QUESTIONS = [
    {"question": "What are the main complaints about credit cards?", "product_filter": None},
    {"question": "Why are customers unhappy with personal loans?", "product_filter": "personal_loan"},
    {"question": "What issues do people have with money transfers?", "product_filter": "money_transfer"},
    {"question": "What are common problems with savings accounts?", "product_filter": "savings_account"},
    {"question": "What billing disputes are customers reporting?", "product_filter": None},
    {"question": "Are there complaints about unauthorized transactions or fraud?", "product_filter": None},
    {"question": "What problems do customers face when trying to close accounts?", "product_filter": None},
    {"question": "What are the most frequent types of complaints across all products?", "product_filter": None},
    {"question": "How do companies typically respond to customer complaints?", "product_filter": None},
    {"question": "What issues are related to fees and interest charges?", "product_filter": None},
]


def run_evaluation():
    """Run evaluation and print results."""
    print("=" * 70)
    print("RAG QUALITATIVE EVALUATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    
    rag = RAGPipeline()
    results = []
    
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {q['question']}")
        if q['product_filter']:
            print(f"Filter: {q['product_filter']}")
        print("-" * 70)
        
        answer, sources = rag.answer(q['question'], product_filter=q['product_filter'])
        
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources ({len(sources)}):")
        for j, src in enumerate(sources, 1):
            print(f"  {j}. [{src['product']}] {src['issue']}")
        
        results.append({
            "question": q['question'],
            "filter": q['product_filter'],
            "answer": answer,
            "num_sources": len(sources),
            "products": list(set(s['product'] for s in sources))
        })
    
    # Summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'#':<3} {'Question':<50} {'Sources':<8} {'Products'}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        q_short = r['question'][:47] + "..." if len(r['question']) > 50 else r['question']
        products = ", ".join(r['products'][:2])
        print(f"{i:<3} {q_short:<50} {r['num_sources']:<8} {products}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_evaluation()
