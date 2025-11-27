import json
import streamlit as st
from coach.services.RAG_service import RAG
from pinecone import Pinecone
from coach.config import VECTOR_DB_KEY, VECTOR_DB_HOST
import time
import ast

# Page configuration
st.set_page_config(
    page_title="AI Career Coach",
    page_icon="💼",
    layout="wide"
)

# Initialize Pinecone and RAG
@st.cache_resource
def initialize_services():
    pc = Pinecone(api_key=VECTOR_DB_KEY)
    index = pc.Index(host=VECTOR_DB_HOST)
    rag = RAG()
    return index, rag

index, rag = initialize_services()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def is_roadmap_format(data):
    """Check if the response is in roadmap format"""
    if isinstance(data, dict):
        return "nodes" in data and "edges" in data and "title" in data
    return False

def format_chat_history(messages):
    """Format chat history for the RAG service"""
    history = []
    for msg in messages:
        # Skip roadmap visualizations in history, just include a reference
        if msg.get("type") == "roadmap":
            history.append({
                "role": msg["role"],
                "content": "[Roadmap visualization was generated]"
            })
        else:
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    return history

def create_roadmap_visualization(roadmap_data):
    """Create an interactive roadmap visualization using HTML and vis.js"""
    title = roadmap_data.get("title", "Learning Roadmap")
    nodes = roadmap_data.get("nodes", [])
    edges = roadmap_data.get("edges", [])
    
    # Convert nodes to vis.js format
    vis_nodes = []
    for node in nodes:
        vis_nodes.append({
            "id": node["id"],
            "label": node["label"],
            "shape": "box",
            "margin": 10,
            "font": {"size": 14}
        })
    
    # Convert edges to vis.js format
    vis_edges = []
    for edge in edges:
        vis_edges.append({
            "from": edge["from"],
            "to": edge["to"],
            "arrows": "to"
        })
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #mynetwork {{
                width: 100%;
                height: 600px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
            }}
            .roadmap-title {{
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #1f77b4;
            }}
        </style>
    </head>
    <body>
        <div class="roadmap-title">{title}</div>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({json.dumps(vis_nodes)});
            var edges = new vis.DataSet({json.dumps(vis_edges)});
            
            var container = document.getElementById('mynetwork');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            var options = {{
                layout: {{
                    hierarchical: {{
                        direction: 'UD',
                        sortMethod: 'directed',
                        levelSeparation: 150,
                        nodeSpacing: 200
                    }}
                }},
                physics: {{
                    enabled: false
                }},
                nodes: {{
                    color: {{
                        border: '#2B7CE9',
                        background: '#D2E5FF',
                        highlight: {{
                            border: '#2B7CE9',
                            background: '#A8D1FF'
                        }}
                    }},
                    font: {{
                        color: '#000000',
                        size: 14
                    }}
                }},
                edges: {{
                    color: {{
                        color: '#848484',
                        highlight: '#2B7CE9'
                    }},
                    smooth: {{
                        type: 'cubicBezier',
                        forceDirection: 'vertical'
                    }}
                }},
                interaction: {{
                    hover: true,
                    zoomView: true,
                    dragView: true
                }}
            }};
            
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    return html_content

# Display header
st.title("💼 AI Career Coach")
st.markdown("Ask me anything about your career!")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "roadmap":
            # Display roadmap visualization
            st.components.v1.html(message["content"], height=700, scrolling=True)
        else:
            # Display normal text message
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about your career?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                start_time = time.time()
                
                # Format chat history for RAG
                chat_history = format_chat_history(st.session_state.messages[:-1])  # Exclude current message
                
                # Get response from RAG with history
                rag._get_user_query(prompt)
                
                # Pass chat history to the LLM call
                # Assuming your RAG service supports a history parameter
                # If not, you may need to modify the RAG service
                try:
                    output = rag._call_llm(index, history=chat_history)
                except TypeError:
                    # Fallback if history parameter is not supported
                    output = rag._call_llm(index)
                
                # Try to parse as JSON or Python dict
                response_data = None
                try:
                    response_data = json.loads(output)
                except json.JSONDecodeError:
                    try:
                        response_data = ast.literal_eval(output)
                    except Exception:
                        response_data = None
                
                end_time = time.time()
                
                # Check if it's a roadmap format
                if response_data and is_roadmap_format(response_data):
                    # Create and display roadmap visualization
                    html_content = create_roadmap_visualization(response_data)
                    st.components.v1.html(html_content, height=700, scrolling=True)
                    
                    # Add roadmap to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": html_content,
                        "type": "roadmap"
                    })
                    
                    st.caption(f"Response time: {end_time - start_time:.2f}s")
                else:
                    # Handle normal message
                    if response_data:
                        if isinstance(response_data, dict) and "message" in response_data:
                            response_text = response_data["message"]
                        else:
                            response_text = json.dumps(response_data, indent=2)
                    else:
                        response_text = output
                    
                    # Display response
                    st.markdown(response_text)
                    st.caption(f"Response time: {end_time - start_time:.2f}s")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

# Sidebar with options
with st.sidebar:
    st.header("Options")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This AI Career Coach uses RAG (Retrieval-Augmented Generation) to provide personalized career advice.")
    st.markdown("- Ask general questions for text responses")
    st.markdown("- Ask for a roadmap to see an interactive visualization")
    st.markdown("- Chat history is maintained during your session")
    
    if st.session_state.messages:
        st.markdown("---")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")