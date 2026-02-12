"""
Production-ready RAG-powered chatbot for analyzing CFPB customer complaints.

Features:
- Input validation and sanitization
- Rate limiting
- Error handling and resilience
- Caching
- Health checks
- Comprehensive logging
- Security best practices

Usage:
    python app.py
"""

import gradio as gr
import traceback
from typing import Tuple, Optional
from src.rag import RAGPipeline
from src.config import (
    APP_HOST, APP_PORT, APP_SHARE, APP_DEBUG, APP_TITLE,
    ENABLE_RATE_LIMITING, RATE_LIMIT_PER_MINUTE, MAX_TOP_K
)
from src.logger import logger, setup_logger
from src.utils import (
    sanitize_input, validate_query, format_error_message,
    rate_limiter
)
from src.health import health_checker
from src.cache import query_cache
from src.metrics import metrics_collector
import time

# Setup logger
setup_logger('complaint_analyzer')

# Initialize RAG pipeline
rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    """Lazy load RAG pipeline with error handling."""
    global rag
    if rag is None:
        try:
            logger.info("Initializing RAG pipeline...")
            rag = RAGPipeline()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            logger.error(traceback.format_exc())
            raise
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


def format_answer(answer: str, is_error: bool = False):
    """Format answer with clean styling."""
    if not answer:
        return ""
    
    if is_error:
        return f"""
        <div style="background: #fef2f2; padding: 24px; border-radius: 8px; border: 1px solid #fecaca;">
            <p style="margin: 0; color: #991b1b; font-size: 15px; line-height: 1.8;">{answer}</p>
        </div>
        """
    
    return f"""
    <div style="background: #ffffff; padding: 24px; border-radius: 8px; border: 1px solid #e2e8f0;">
        <p style="margin: 0; color: #1e293b; font-size: 15px; line-height: 1.8;">{answer}</p>
    </div>
    """


def analyze(
    message: str,
    product_filter: str,
    num_sources: int,
    request: gr.Request = None
) -> Tuple[str, str]:
    """Process user message and return response with sources.
    
    Args:
        message: User's question
        product_filter: Product category filter
        num_sources: Number of sources to retrieve
        request: Gradio request object (for rate limiting)
        
    Returns:
        Tuple of (formatted_answer, formatted_sources)
    """
    try:
        # Validate input
        if not message or not message.strip():
            return format_answer("Please enter a question to begin analysis."), ""
        
        # Sanitize input
        message = sanitize_input(message)
        
        # Validate query
        is_valid, error_msg = validate_query(message)
        if not is_valid:
            logger.warning(f"Invalid query rejected: {error_msg}")
            return format_answer(f"Invalid input: {error_msg}", is_error=True), ""
        
        # Rate limiting
        if ENABLE_RATE_LIMITING and request:
            client_ip = getattr(request, 'client', {}).get('host', 'unknown')
            if not rate_limiter.is_allowed(client_ip):
                remaining = rate_limiter.get_remaining(client_ip)
                error_msg = (
                    f"Rate limit exceeded. Please wait before making another request. "
                    f"Limit: {RATE_LIMIT_PER_MINUTE} requests per minute."
                )
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return format_answer(error_msg, is_error=True), ""
        
        # Clamp num_sources
        num_sources = min(max(3, int(num_sources)), MAX_TOP_K)
        
        # Map product filter
        filter_map = {
            "All Products": None,
            "Credit Card": "credit_card",
            "Personal Loan": "personal_loan",
            "Savings Account": "savings_account",
            "Money Transfer": "money_transfer"
        }
        
        # Get RAG pipeline
        try:
            pipeline = get_rag()
        except Exception as e:
            logger.error(f"Failed to get RAG pipeline: {e}")
            return format_answer(
                "System initialization error. Please check logs and ensure all services are running.",
                is_error=True
            ), ""
        
        # Check cache first
        cache_key = f"{message}|{product_filter}|{num_sources}"
        cached_result = query_cache.get(cache_key)
        start_time = time.time()
        
        if cached_result:
            logger.info("Returning cached result")
            duration = time.time() - start_time
            metrics_collector.record_query(duration, cached=True)
            return format_answer(cached_result['answer']), format_sources(cached_result['sources'])
        
        # Process query
        logger.info(f"Processing query: {message[:100]}...")
        
        try:
            answer, sources = pipeline.answer(
                question=message,
                product_filter=filter_map.get(product_filter),
                top_k=num_sources,
                use_cache=False  # We handle caching here
            )
            
            # Cache result
            query_cache.set(cache_key, {
                'answer': answer,
                'sources': sources
            })
            
            duration = time.time() - start_time
            logger.info(f"Query processed successfully in {duration:.2f}s. Retrieved {len(sources)} sources")
            
            # Record metrics
            metrics_collector.record_query(duration, cached=False)
            
            return format_answer(answer), format_sources(sources)
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            metrics_collector.record_error(error_type)
            raise
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        error_msg = format_error_message(e, include_details=APP_DEBUG)
        return format_answer(error_msg, is_error=True), ""


def get_health_status() -> str:
    """Get system health status for display."""
    try:
        health = health_checker.overall_health()
        status = health['status']
        uptime = health['uptime_seconds']
        
        status_html = f"""
        <div style="background: {'#f0fdf4' if status == 'healthy' else '#fef2f2'}; padding: 16px; border-radius: 8px; border: 1px solid {'#86efac' if status == 'healthy' else '#fecaca'};">
            <h3 style="margin: 0 0 8px 0; color: {'#166534' if status == 'healthy' else '#991b1b'};">
                System Status: {status.upper()}
            </h3>
            <p style="margin: 0; color: #64748b; font-size: 14px;">
                Uptime: {uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s
            </p>
        </div>
        """
        return status_html
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return "<p style='color: #991b1b;'>Health check unavailable</p>"


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
with gr.Blocks(css=custom_css, title=APP_TITLE) as demo:
    
    # Header
    gr.HTML(f"""
        <div class="main-header">
            <h1>{APP_TITLE}</h1>
            <p>Intelligent analysis of customer complaints powered by RAG</p>
        </div>
    """)
    
    # Health status (optional, can be hidden)
    if APP_DEBUG:
        with gr.Accordion("System Status", open=False):
            health_status = gr.HTML(value=get_health_status())
            refresh_health_btn = gr.Button("Refresh Status", size="sm")
            refresh_health_btn.click(
                fn=get_health_status,
                outputs=health_status
            )
    
    # Input Section
    with gr.Group(elem_classes="input-section"):
        question_input = gr.Textbox(
            label="Question",
            placeholder="What patterns or issues would you like to analyze?",
            lines=2,
            show_label=True,
            max_lines=5
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
                maximum=MAX_TOP_K,
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
    logger.info("=" * 60)
    logger.info(f"Starting {APP_TITLE}")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Host: {APP_HOST}")
    logger.info(f"  Port: {APP_PORT}")
    logger.info(f"  Debug: {APP_DEBUG}")
    logger.info(f"  Rate limiting: {ENABLE_RATE_LIMITING}")
    
    # Pre-initialize RAG pipeline
    try:
        logger.info("Pre-initializing RAG pipeline...")
        get_rag()
        logger.info("RAG pipeline ready")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        logger.error("Application will start but queries may fail")
    
    # Health check
    try:
        health = health_checker.overall_health()
        logger.info(f"System health: {health['status']}")
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
    
    logger.info("Starting Gradio interface...")
    demo.launch(
        server_name=APP_HOST,
        server_port=APP_PORT,
        share=APP_SHARE,
        show_error=APP_DEBUG
    )
