import streamlit as st
import os
import tempfile
from workflows.upsert_workflow import RAGSystem
from workflows.retrive_workflow import RAGRetriever

def main():
    print("Hello from rag-project!")


if __name__ == "__main__":
    main()
    def create_streamlit_app():
        st.set_page_config(page_title="RAG Chat System", page_icon="🤖", layout="wide")
        
        st.title("🤖 RAG Document Chat System")
        st.markdown("Upload PDF or TXT files and chat with your documents!")
        
        # Sidebar for file upload
        with st.sidebar:
            st.header("📁 Document Upload")
            uploaded_files = st.file_uploader(
                "Choose files", 
                type=['pdf', 'txt'], 
                accept_multiple_files=True,
                help="Upload PDF or TXT files to add to the knowledge base"
            )
            
            if st.button("Process Documents", type="primary"):
                if uploaded_files:
                    with st.spinner("Processing documents..."):
                        # Create temporary directory for uploaded files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save uploaded files to temp directory
                            for uploaded_file in uploaded_files:
                                file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                            
                            # Process documents using upsert workflow
                            try:
                                rag_system = RAGSystem(temp_dir)
                                rag_system.workflow()
                                st.success(f"Successfully processed {len(uploaded_files)} documents!")
                            except Exception as e:
                                st.error(f"Error processing documents: {str(e)}")
                else:
                    st.warning("Please upload files first!")
        
        # Main chat interface
        st.header("💬 Chat with Documents")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Initialize retriever
        if "retriever" not in st.session_state:
            try:
                st.session_state.retriever = RAGRetriever()
            except Exception as e:
                st.error(f"Error initializing retriever: {str(e)}")
                return
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response using retrieve workflow
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.retriever.ask(prompt)
                        response = result["response"]
                        st.markdown(response)
                        
                        # Show retrieved documents in an expander
                        if result["retrieved_documents"]:
                            with st.expander(f"📚 View {result['num_retrieved']} source documents"):
                                for i, doc in enumerate(result["retrieved_documents"], 1):
                                    st.markdown(f"**Document {i}** (Relevance: {doc['relevance_score']:.3f})")
                                    st.text(doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"])
                                    st.markdown("---")
                        
                    except Exception as e:
                        response = f"Sorry, I encountered an error: {str(e)}"
                        st.error(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # System info in sidebar
        with st.sidebar:
            st.markdown("---")
            st.header("📊 System Info")
            if st.button("Show System Status"):
                try:
                    info = st.session_state.retriever.get_system_info()
                    st.json(info)
                except Exception as e:
                    st.error(f"Error getting system info: {str(e)}")

    if __name__ == "__main__":
        create_streamlit_app()