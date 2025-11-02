"""Chat module.

This module handles the chat interface and response generation
using the OpenAI API.
"""

from openai import OpenAI
import streamlit as st


client = OpenAI()


def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.

IMPORTANT: Always extract and list specific technical parameters, settings, and values from the context.
When asked about configuration parameters (like kernel parameters, system settings, etc.), you MUST:
1. Identify ALL relevant parameters in the context
2. List them clearly with their exact values
3. Include the source information (file name and page numbers)

If the context contains kernel parameters like "net.ipv4.tcp_rmem = 4194304 8388608 25165824", you MUST include this exact parameter in your answer.

Use only the information from the context to answer questions. If you're unsure or the context
doesn't contain the relevant information, say so.

Context:
{context}
    """

    messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_context,
        temperature=0.5,
        stream=True,
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


def display_search_results(context: str):
    """Display search results with expandable sections.

    Args:
        context: Concatenated context string from get_context
    """
    st.markdown(
        """
        <style>
        .search-result {
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
            background-color: #f0f2f6;
        }
        .search-result summary {
            cursor: pointer;
            color: #0f52ba;
            font-weight: 500;
        }
        .search-result summary:hover {
            color: #1e90ff;
        }
        .metadata {
            font-size: 0.9em;
            color: #666;
            font-style: italic;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.write("Found relevant sections:")
    for chunk in context.split("\n\n"):
        # Split into text and metadata parts
        parts = chunk.split("\n")
        text = parts[0]
        metadata = {
            line.split(": ")[0]: line.split(": ")[1]
            for line in parts[1:]
            if ": " in line
        }

        source = metadata.get("Source", "Unknown source")
        title = metadata.get("Title", "Untitled section")

        st.markdown(
            f"""
            <div class="search-result">
                <details>
                    <summary>{source}</summary>
                    <div class="metadata">Section: {title}</div>
                    <div style="margin-top: 8px;">{text}</div>
                </details>
            </div>
        """,
            unsafe_allow_html=True,
        )


def run_chat_interface(table):
    """Run the main chat interface.

    Args:
        table: LanceDB table object
    """
    st.title("📚 Document Q&A")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get relevant context
        with st.status("Searching document...", expanded=False) as status:
            from query import get_context
            context = get_context(prompt, table)
            display_search_results(context)

        # Display assistant response first
        with st.chat_message("assistant"):
            # Get model response with streaming
            response = get_chat_response(st.session_state.messages, context)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
