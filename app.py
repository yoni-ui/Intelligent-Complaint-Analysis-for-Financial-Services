"""
Task 4: Interactive Chat Interface using Gradio

A RAG-powered chatbot for analyzing CFPB customer complaints.

Usage:
    python app.py
"""

import gradio as gr
from src.rag import RAGPipeline

# Initialize RAG pipeline
rag = None

def get_rag():
    """Lazy load RAG pipeline."""
    global rag
    if rag is None:
        rag = RAGPipeline()
    return rag


def format_sources(sources):
    """Format sources for display."""
    if not sources:
        return "<p style='color: #6b7280;'>No sources found.</p>"
    
    html_parts = []
    for i, src in enumerate(sources, 1):
        text_preview = src['text'][:280] + "..." if len(src['text']) > 280 else src['text']
        html_parts.append(f"""
        <div style="background: #f8fafc; border-left: 3px solid #3b82f6; padding: 16px; margin-bottom: 12px; border-radius: 4px;">
            <div style="display: flex; gap: 16px; margin-bottom: 8px; font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                <span>Source {i}</span>
                <span style="color: #3b82f6;">{src['product'].replace('_', ' ').title()}</span>
                <span>{src['issue']}</span>
            </div>
            <p style="margin: 0; color: #334155; font-size: 14px; line-height: 1.6;">{text_preview}</p>
        </div>
        """)
    return "".join(html_parts)


def format_answer(answer):
    """Format answer with clean styling."""
    if not answer:
        return ""
    return f"""
    <div style="background: #ffffff; padding: 24px; border-radius: 8px; border: 1px solid #e2e8f0;">
        <p style="margin: 0; color: #1e293b; font-size: 15px; line-height: 1.8;">{answer}</p>
    </div>
    """


def analyze(message, product_filter, num_sources):
    """Process user message and return response with sources."""
    if not message.strip():
        return "<p style='color: #94a3b8;'>Enter a question to begin analysis.</p>", ""
    
    filter_map = {
        "All Products": None,
        "Credit Card": "credit_card",
        "Personal Loan": "personal_loan",
        "Savings Account": "savings_account",
        "Money Transfer": "money_transfer"
    }
    
    pipeline = get_rag()
    answer, sources = pipeline.answer(
        question=message,
        product_filter=filter_map.get(product_filter),
        top_k=int(num_sources)
    )
    
    return format_answer(answer), format_sources(sources)


# Custom CSS for modern, minimalist design
custom_css = """
:root {
    --primary: #0f172a;
    --secondary: #3b82f6;
    --bg: #ffffff;
    --surface: #f8fafc;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-muted: #64748b;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.main-header {
    text-align: center;
    padding: 48px 24px 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}

.main-header h1 {
    font-size: 28px;
    font-weight: 600;
    color: var(--primary);
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

.main-header p {
    font-size: 15px;
    color: var(--text-muted);
    margin: 0;
}

.input-section {
    background: var(--surface);
    padding: 24px;
    border-radius: 12px;
    border: 1px solid var(--border);
}

.results-section {
    margin-top: 24px;
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
}

footer {
    display: none !important;
}

.gr-button-primary {
    background: var(--primary) !important;
    border: none !important;
    font-weight: 500 !important;
}

.gr-button-primary:hover {
    background: #1e293b !important;
}

.example-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-size: 13px !important;
}

.example-btn:hover {
    background: var(--surface) !important;
    border-color: var(--secondary) !important;
}
"""

# Build interface
with gr.Blocks(css=custom_css, title="CrediTrust Complaint Analyzer") as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>CrediTrust Complaint Analyzer</h1>
            <p>Intelligent analysis of customer complaints powered by RAG</p>
        </div>
    """)
    
    # Input Section
    with gr.Group(elem_classes="input-section"):
        question_input = gr.Textbox(
            label="Question",
            placeholder="What patterns or issues would you like to analyze?",
            lines=2,
            show_label=True
        )
        
        with gr.Row():
            product_filter = gr.Dropdown(
                choices=["All Products", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"],
                value="All Products",
                label="Product Filter",
                scale=2
            )
            num_sources = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="Sources",
                scale=1
            )
            submit_btn = gr.Button("Analyze", variant="primary", scale=1)
    
    # Quick Examples
    gr.HTML("<p class='section-label' style='margin-top: 24px;'>Quick queries</p>")
    with gr.Row():
        ex1 = gr.Button("Credit card complaints", size="sm", elem_classes="example-btn")
        ex2 = gr.Button("Billing disputes", size="sm", elem_classes="example-btn")
        ex3 = gr.Button("Unauthorized transactions", size="sm", elem_classes="example-btn")
        ex4 = gr.Button("Account closure issues", size="sm", elem_classes="example-btn")
    
    # Results Section
    gr.HTML("<p class='section-label' style='margin-top: 32px;'>Analysis</p>")
    answer_output = gr.HTML(elem_classes="results-section")
    
    gr.HTML("<p class='section-label' style='margin-top: 24px;'>Source Documents</p>")
    sources_output = gr.HTML()
    
    # Event handlers
    submit_btn.click(
        fn=analyze,
        inputs=[question_input, product_filter, num_sources],
        outputs=[answer_output, sources_output]
    )
    
    question_input.submit(
        fn=analyze,
        inputs=[question_input, product_filter, num_sources],
        outputs=[answer_output, sources_output]
    )
    
    # Example handlers
    ex1.click(lambda: "What are the main complaints about credit cards?", outputs=question_input)
    ex2.click(lambda: "What billing disputes are customers reporting?", outputs=question_input)
    ex3.click(lambda: "Are there complaints about unauthorized transactions or fraud?", outputs=question_input)
    ex4.click(lambda: "What problems do customers face when trying to close accounts?", outputs=question_input)


if __name__ == "__main__":
    print("Starting CrediTrust Complaint Analyzer...")
    print("Loading RAG pipeline...")
    get_rag()
    print("Ready.")
    demo.launch(share=False)
