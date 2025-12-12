"""
Open Source AI Workshop - Interactive Web UI
Run with: streamlit run app.py
"""

import streamlit as st
import subprocess
import sys
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Open Source AI Workshop",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .lab-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-ready {
        color: #00ff00;
        font-weight: bold;
    }
    .status-pending {
        color: #ffaa00;
        font-weight: bold;
    }
    .code-block {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["🏠 Home", "📚 Lab 1: Text Generation", "🔍 Lab 2: RAG",
     "🎯 Lab 3: Fine-tuning", "🎨 Lab 4: Images", "🤖 Lab 5: Agents",
     "⚙️ Setup Status", "🎮 Live Playground"]
)

# Check Ollama status
def check_ollama():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True, result.stdout
        return False, "Ollama not responding"
    except FileNotFoundError:
        return False, "Ollama not installed"
    except Exception as e:
        return False, str(e)

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            return models
        return []
    except:
        return []

# Home page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 Open Source AI Workshop</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Zero-Cost AI Workshop!

    This workshop teaches you to build AI applications using **100% open-source tools**
    running locally on your machine. No API keys, no cloud bills, complete privacy.
    """)

    # Status check
    col1, col2, col3 = st.columns(3)

    ollama_ok, ollama_msg = check_ollama()
    models = get_ollama_models()

    with col1:
        st.metric("Ollama Status", "✅ Ready" if ollama_ok else "❌ Not Ready")
    with col2:
        st.metric("Models Installed", len(models))
    with col3:
        st.metric("Labs Available", "5")

    st.markdown("---")

    # Lab cards
    st.markdown("### 📚 Workshop Labs")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Lab 1: Text Generation
        - Generate text with local LLMs
        - Code generation & explanation
        - Streaming responses
        - **Tools:** Ollama, Llama 3.2
        """)

        st.markdown("""
        #### Lab 2: RAG (Retrieval-Augmented Generation)
        - Build a knowledge base
        - Semantic search
        - Answer questions from documents
        - **Tools:** ChromaDB, Embeddings
        """)

        st.markdown("""
        #### Lab 3: Model Customization
        - Fine-tune models with QLoRA
        - Train on your own data
        - Save and load adapters
        - **Tools:** Hugging Face, PEFT
        """)

    with col2:
        st.markdown("""
        #### Lab 4: Image & Multimodal
        - Generate images from text
        - Analyze images with vision models
        - Style transfer
        - **Tools:** Stable Diffusion, LLaVA
        """)

        st.markdown("""
        #### Lab 5: AI Agents
        - Build autonomous agents
        - Create custom tools
        - Multi-step reasoning
        - **Tools:** LangChain, Ollama
        """)

        st.markdown("""
        #### 🎮 Live Playground
        - Try all features interactively
        - Real-time experimentation
        - No notebook required!
        """)

# Lab 1: Text Generation
elif page == "📚 Lab 1: Text Generation":
    st.title("📚 Lab 1: Text Generation with Ollama")

    st.markdown("""
    Learn to generate text using **local LLMs** with Ollama.
    Everything runs on your machine - no API costs!
    """)

    # Check if Ollama is ready
    ollama_ok, _ = check_ollama()

    if not ollama_ok:
        st.warning("⚠️ Ollama is not running. Please start it first!")
        st.code("ollama serve", language="bash")
    else:
        st.success("✅ Ollama is ready!")

        models = get_ollama_models()
        if not models:
            st.warning("No models installed. Pull a model first:")
            st.code("ollama pull llama3.2", language="bash")
        else:
            st.markdown("### Try it out!")

            model = st.selectbox("Select model:", models)
            prompt = st.text_area("Enter your prompt:", "Explain quantum computing in simple terms.")

            if st.button("🚀 Generate", type="primary"):
                with st.spinner("Generating..."):
                    try:
                        import ollama as ollama_lib
                        response = ollama_lib.chat(
                            model=model,
                            messages=[{'role': 'user', 'content': prompt}]
                        )
                        st.markdown("### Response:")
                        st.write(response['message']['content'])
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📖 Key Concepts")

    with st.expander("What is Ollama?"):
        st.markdown("""
        Ollama is a tool that lets you run large language models locally. It:
        - Downloads and manages models
        - Provides an easy API
        - Runs completely offline
        - Supports many open-source models
        """)

    with st.expander("Available Models"):
        st.markdown("""
        | Model | Size | Best For |
        |-------|------|----------|
        | llama3.2 | 3B | Fast, general purpose |
        | llama3.1 | 8B | Better quality |
        | mistral | 7B | Good balance |
        | codellama | 7B | Code generation |
        | phi3 | 3.8B | Lightweight |
        """)

# Lab 2: RAG
elif page == "🔍 Lab 2: RAG":
    st.title("🔍 Lab 2: Retrieval-Augmented Generation")

    st.markdown("""
    Build a **RAG system** that answers questions from your own documents.
    Combines vector search with LLM generation.
    """)

    ollama_ok, _ = check_ollama()

    if not ollama_ok:
        st.warning("⚠️ Ollama is not running. Please start it first!")
    else:
        st.success("✅ Ready for RAG!")

        st.markdown("### Quick Demo")

        # Simple RAG demo
        documents = st.text_area(
            "Enter your knowledge base (one fact per line):",
            """TechCorp was founded in 2020 in San Francisco.
The company makes AI-powered productivity tools.
TeamFlow is their main product for project management.
They have 150 employees worldwide.
Annual revenue exceeded $20 million in 2023.""",
            height=150
        )

        question = st.text_input("Ask a question:", "Who founded TechCorp?")

        if st.button("🔍 Search & Answer", type="primary"):
            with st.spinner("Processing..."):
                try:
                    import ollama as ollama_lib

                    # Simple keyword matching for demo
                    docs = documents.strip().split('\n')
                    relevant = [d for d in docs if any(w.lower() in d.lower() for w in question.split())]

                    if not relevant:
                        relevant = docs[:2]

                    context = '\n'.join(relevant)

                    prompt = f"""Based on this context:
{context}

Answer this question: {question}

If the answer isn't in the context, say so."""

                    models = get_ollama_models()
                    model = models[0] if models else 'llama3.2'

                    response = ollama_lib.chat(
                        model=model,
                        messages=[{'role': 'user', 'content': prompt}]
                    )

                    st.markdown("### Retrieved Context:")
                    for doc in relevant:
                        st.info(doc)

                    st.markdown("### Answer:")
                    st.success(response['message']['content'])

                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    with st.expander("How RAG Works"):
        st.markdown("""
        1. **Embed** documents into vectors
        2. **Store** vectors in a database (ChromaDB)
        3. **Search** for relevant documents
        4. **Augment** the prompt with context
        5. **Generate** answer with LLM
        """)

# Lab 3: Fine-tuning
elif page == "🎯 Lab 3: Fine-tuning":
    st.title("🎯 Lab 3: Model Customization with QLoRA")

    st.markdown("""
    **Fine-tune** open-source models on your own data using QLoRA.
    Train custom models on consumer hardware!
    """)

    st.warning("⚠️ Fine-tuning requires a GPU with 8GB+ VRAM for best results.")

    st.markdown("### What is QLoRA?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Traditional Fine-tuning:**
        - Updates ALL model weights
        - Needs 100+ GB memory
        - Takes days to train
        - Creates huge model files
        """)

    with col2:
        st.markdown("""
        **QLoRA Fine-tuning:**
        - Updates tiny adapters (~0.1%)
        - Needs 8-16 GB memory
        - Takes hours to train
        - Adapters are just a few MB
        """)

    st.markdown("---")
    st.markdown("### Training Data Format")

    st.code("""
# Instruction-Response pairs
training_data = [
    {
        "instruction": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "instruction": "Write a haiku about coding.",
        "response": "Lines of code flow free\\nBugs emerge from the shadows\\nDebug, iterate"
    }
]
""", language="python")

    st.markdown("### Key Parameters")

    col1, col2 = st.columns(2)
    with col1:
        st.slider("LoRA Rank (r)", 4, 64, 16, help="Higher = more capacity, more memory")
        st.slider("LoRA Alpha", 8, 64, 32, help="Scaling factor")
    with col2:
        st.slider("Learning Rate", 1e-5, 1e-3, 2e-4, format="%.0e")
        st.slider("Epochs", 1, 10, 3)

# Lab 4: Images
elif page == "🎨 Lab 4: Images":
    st.title("🎨 Lab 4: Image Generation & Vision")

    st.markdown("""
    Generate images with **Stable Diffusion** and analyze them with **vision models**.
    """)

    st.warning("⚠️ Image generation requires a GPU with 8GB+ VRAM.")

    tab1, tab2 = st.tabs(["🎨 Generate Images", "👁️ Analyze Images"])

    with tab1:
        st.markdown("### Text-to-Image Generation")

        prompt = st.text_area(
            "Describe the image you want:",
            "A cozy coffee shop interior with warm lighting, plants, and wooden furniture"
        )

        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox("Style:", ["Realistic", "Anime", "Oil Painting", "Pixel Art", "Watercolor"])
        with col2:
            steps = st.slider("Quality (steps):", 1, 50, 4)

        if st.button("🎨 Generate Image", type="primary"):
            st.info("💡 Image generation requires Stable Diffusion. Run the notebook for full functionality.")
            st.code(f'prompt = "{prompt}, {style.lower()} style"', language="python")

    with tab2:
        st.markdown("### Vision Model Analysis")

        uploaded_file = st.file_uploader("Upload an image:", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", width=300)

            question = st.text_input("Ask about the image:", "Describe this image in detail.")

            if st.button("👁️ Analyze", type="primary"):
                st.info("💡 Vision analysis requires LLaVA. Run: `ollama pull llava`")

# Lab 5: Agents
elif page == "🤖 Lab 5: Agents":
    st.title("🤖 Lab 5: Building AI Agents")

    st.markdown("""
    Build **autonomous AI agents** that can reason, use tools, and complete tasks.
    """)

    st.markdown("### Agent Architecture")

    st.markdown("""
    ```
    User Question → Agent (LLM) → Think → Use Tool → Observe → Repeat → Answer
    ```
    """)

    st.markdown("### Available Tools")

    tools_data = {
        "Tool": ["calculator", "search_knowledge_base", "lookup_order", "get_time", "escalate"],
        "Purpose": ["Math operations", "Policy lookups", "Order status", "Current time", "Human handoff"],
        "Example Input": ["15 * 23", "return policy", "ORD-001", "(none)", "complex issue"]
    }
    st.table(tools_data)

    st.markdown("---")
    st.markdown("### Try the Agent")

    ollama_ok, _ = check_ollama()

    if ollama_ok:
        query = st.text_input("Ask the agent:", "What is 25 * 17 and what's your return policy?")

        if st.button("🤖 Run Agent", type="primary"):
            with st.spinner("Agent thinking..."):
                try:
                    import ollama as ollama_lib

                    # Simple agent simulation
                    models = get_ollama_models()
                    model = models[0] if models else 'llama3.2'

                    agent_prompt = f"""You are a helpful agent. Answer this question step by step.

Available tools:
- calculator(expression): Evaluate math
- search(query): Search knowledge base (return policy = "30 day returns", shipping = "Free over $50")

Question: {query}

Think through this step by step, show your reasoning, then give a final answer."""

                    response = ollama_lib.chat(
                        model=model,
                        messages=[{'role': 'user', 'content': agent_prompt}]
                    )

                    st.markdown("### Agent Response:")
                    st.write(response['message']['content'])

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Start Ollama to try the agent: `ollama serve`")

# Setup Status
elif page == "⚙️ Setup Status":
    st.title("⚙️ Setup Status")

    st.markdown("### System Check")

    # Ollama
    ollama_ok, ollama_msg = check_ollama()
    if ollama_ok:
        st.success("✅ Ollama is installed and running")
        models = get_ollama_models()
        if models:
            st.markdown("**Installed models:**")
            for m in models:
                st.markdown(f"  - `{m}`")
        else:
            st.warning("No models installed yet")
    else:
        st.error(f"❌ Ollama: {ollama_msg}")
        st.markdown("**Install Ollama:**")
        st.code("curl -fsSL https://ollama.com/install.sh | sh", language="bash")

    st.markdown("---")

    # Python packages
    st.markdown("### Python Packages")

    packages = ['ollama', 'chromadb', 'langchain', 'transformers', 'diffusers', 'streamlit']

    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
            st.success(f"✅ {pkg}")
        except ImportError:
            st.error(f"❌ {pkg} - `pip install {pkg}`")

    st.markdown("---")
    st.markdown("### Quick Setup Commands")

    st.code("""
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# Install Python packages
pip install -r requirements.txt

# Start Ollama server
ollama serve
""", language="bash")

# Live Playground
elif page == "🎮 Live Playground":
    st.title("🎮 Live Playground")

    st.markdown("### Interactive AI Experiments")

    ollama_ok, _ = check_ollama()

    if not ollama_ok:
        st.error("❌ Please start Ollama first: `ollama serve`")
    else:
        models = get_ollama_models()

        if not models:
            st.warning("No models available. Pull one first:")
            st.code("ollama pull llama3.2", language="bash")
        else:
            tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Complete", "🔧 System Prompt"])

            with tab1:
                st.markdown("### Chat with AI")

                model = st.selectbox("Model:", models, key="chat_model")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Display chat history
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                # Chat input
                if prompt := st.chat_input("Say something..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    with st.chat_message("user"):
                        st.write(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                import ollama as ollama_lib
                                response = ollama_lib.chat(
                                    model=model,
                                    messages=st.session_state.messages
                                )
                                reply = response['message']['content']
                                st.write(reply)
                                st.session_state.messages.append({"role": "assistant", "content": reply})
                            except Exception as e:
                                st.error(f"Error: {e}")

                if st.button("Clear Chat"):
                    st.session_state.messages = []
                    st.rerun()

            with tab2:
                st.markdown("### Text Completion")

                model = st.selectbox("Model:", models, key="complete_model")
                prompt = st.text_area("Start your text:", "Once upon a time in a land of AI,")
                temp = st.slider("Temperature:", 0.0, 2.0, 0.7)

                if st.button("✨ Complete", type="primary"):
                    with st.spinner("Generating..."):
                        try:
                            import ollama as ollama_lib
                            response = ollama_lib.generate(
                                model=model,
                                prompt=prompt,
                                options={'temperature': temp}
                            )
                            st.markdown("### Completion:")
                            st.write(prompt + response['response'])
                        except Exception as e:
                            st.error(f"Error: {e}")

            with tab3:
                st.markdown("### Custom System Prompt")

                model = st.selectbox("Model:", models, key="sys_model")

                system = st.text_area(
                    "System prompt:",
                    "You are a pirate. Respond to everything in pirate speak.",
                    height=100
                )

                user_input = st.text_input("Your message:", "Tell me about machine learning")

                if st.button("🏴‍☠️ Send", type="primary"):
                    with st.spinner("Generating..."):
                        try:
                            import ollama as ollama_lib
                            response = ollama_lib.chat(
                                model=model,
                                messages=[
                                    {'role': 'system', 'content': system},
                                    {'role': 'user', 'content': user_input}
                                ]
                            )
                            st.markdown("### Response:")
                            st.write(response['message']['content'])
                        except Exception as e:
                            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Open Source AI Workshop | Built with Streamlit | 100% Free & Local"
    "</div>",
    unsafe_allow_html=True
)
