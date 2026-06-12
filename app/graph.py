from typing import List, TypedDict, Dict, Any
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .llm import get_llm
from .vectorstore import get_vectorstore

class GraphState(TypedDict):
    question: str
    chat_history: List[dict]
    query_type: str
    search_query: str
    generation: str
    documents: List[Document]
    retries: int

class QueryAnalysisResult(BaseModel):
    query_type: str = Field(description="Query category: 'conceptual', 'how-to', 'troubleshooting', 'api_reference', or 'other'")
    search_query: str = Field(description="Expanded/rewritten version of the query to optimize for vector search (adding synonyms, resolving terminology)")

class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class HallucinationGrade(BaseModel):
    binary_score: str = Field(description="Is the answer supported by/grounded in the documents? 'yes' or 'no'")

def analyze_query(state: GraphState):
    """Analyze the user's raw question, classify it, and expand/rewrite it for retrieval."""
    print("---ANALYZE QUERY---")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    retries = state.get("retries", 0)
    
    llm = get_llm(temperature=0)
    
    # If we have conversation memory, formulate a standalone question
    standalone_question = question
    if chat_history:
        print(f"Formulating standalone question incorporating {len(chat_history)} chat history turns...")
        history_str = ""
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"{role.capitalize()}: {content}\n"
            
        system_condense = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be answered independently of the history, in English. Do not answer the question, just rephrase it."""
        condense_prompt = ChatPromptTemplate.from_messages([
            ("system", system_condense),
            ("human", f"Chat History:\n{history_str}\n\nFollow-up Question: {question}")
        ])
        try:
            condense_chain = condense_prompt | llm
            standalone_question = condense_chain.invoke({}).content
            print(f"Formulated Standalone Question: {standalone_question}")
        except Exception as e:
            print(f"Error condensing history: {e}. Using raw question.")
            standalone_question = question
            
    system = """You are a technical query analyst. Analyze the user's question about technical libraries (like FastAPI).
Classify it into one of these categories:
- 'conceptual': explanation of a concept
- 'how-to': steps to achieve a specific goal
- 'troubleshooting': debugging an error
- 'api_reference': specific functions, classes, arguments
- 'other': anything else

Also, generate an expanded search query that includes synonyms, key technical terms, or resolves ambiguities to improve similarity search in a vector database."""
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}"),
    ])
    
    try:
        structured_llm = llm.with_structured_output(QueryAnalysisResult)
        analyzer = analysis_prompt | structured_llm
        result = analyzer.invoke({"question": standalone_question})
        query_type = result.query_type
        search_query = result.search_query
    except Exception as e:
        print(f"Error in structured query analysis: {e}. Falling back to plain text parsing.")
        try:
            # Plain text fallback
            raw_analyzer = analysis_prompt | llm
            raw_response = raw_analyzer.invoke({"question": standalone_question}).content
            query_type = "other"
            search_query = standalone_question
            # Simple keyword matching from raw response if structured call failed
            for cat in ['conceptual', 'how-to', 'troubleshooting', 'api_reference']:
                if cat in raw_response.lower():
                    query_type = cat
                    break
        except Exception as ex:
            print(f"Fallback query analysis also failed: {ex}")
            query_type = "other"
            search_query = standalone_question
        
    print(f"Query Category: {query_type}")
    print(f"Expanded Search Query: {search_query}")
    
    return {"query_type": query_type, "search_query": search_query, "retries": retries}


def retrieve(state: GraphState):
    """Retrieve documents from the vector store using the search query."""
    question = state["question"]
    search_query = state.get("search_query", question)
    print(f"---RETRIEVE--- Query: {search_query}")
    vectorstore = get_vectorstore()
    
    try:
        documents = vectorstore.similarity_search(search_query, k=4)
        print(f"Retrieved {len(documents)} document chunks.")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        documents = []
        
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        print("No documents to grade.")
        return {"documents": [], "question": question}
        
    llm = get_llm(temperature=0)
    
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    filtered_docs = []
    for d in documents:
        try:
            structured_llm_grader = llm.with_structured_output(Grade)
            retrieval_grader = grade_prompt | structured_llm_grader
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
        except Exception as e:
            print(f"Error grading document with structured output: {e}. Falling back to plain text parsing.")
            try:
                raw_grader = grade_prompt | llm
                raw_response = raw_grader.invoke({"question": question, "document": d.page_content}).content
                grade = "no" if "no" in raw_response.lower() and "yes" not in raw_response.lower() else "yes"
            except Exception as ex:
                print(f"Fallback grading also failed: {ex}")
                grade = "yes"  # Default to keeping document in case of complete API failure
                
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            
    return {"documents": filtered_docs, "question": question}

def rewrite_query(state: GraphState):
    """Rewrite the search query to improve retrieval, and increment retry count."""
    print("---REWRITE QUERY---")
    question = state["question"]
    retries = state.get("retries", 0) + 1
    
    llm = get_llm()
    system = """You are a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    try:
        question_rewriter = re_write_prompt | llm
        response = question_rewriter.invoke({"question": question})
        new_query = response.content
    except Exception as e:
        print(f"Error rewriting query: {e}")
        new_query = question
        
    print(f"Rewritten Query: {new_query} | Retry Count: {retries}")
    return {"search_query": new_query, "retries": retries}

def web_search(state: GraphState):
    """Perform a web search fallback using DuckDuckGo to obtain relevant context."""
    print("---WEB SEARCH FALLBACK---")
    question = state["question"]
    search_query = state.get("search_query", question)
    
    web_docs = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
            
        if results:
            content_parts = []
            for r in results:
                title = r.get("title", "No Title")
                href = r.get("href", "No Link")
                body = r.get("body", "")
                content_parts.append(f"Title: {title}\nURL: {href}\nContent: {body}")
            
            web_content = "\n\n---\n\n".join(content_parts)
            web_docs = [Document(page_content=web_content, metadata={"source": "DuckDuckGo Web Search"})]
            print(f"Web search found {len(results)} results.")
        else:
            print("Web search returned no results.")
    except Exception as e:
        print(f"Error executing web search: {e}")
        
    return {"documents": web_docs, "question": question}

def generate(state: GraphState):
    """Generate final answer grounded in context."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"generation": "I'm sorry, I could not find any relevant documentation context or web search results to answer your question."}
    
    llm = get_llm(temperature=0)
    context = "\n\n".join(doc.page_content for doc in documents)
    sources = "\n".join(set([doc.metadata.get("source", "Unknown Source") for doc in documents]))
    
    system = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Do not make up facts. Ground your response in the provided context. Include citations or refer to the source files/URLs in the text when explaining."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context: \n\n {context} \n\n Question: {question}"),
    ])
    
    try:
        rag_chain = prompt | llm
        response = rag_chain.invoke({"context": context, "question": question}).content
    except Exception as e:
        print(f"Error generating answer: {e}")
        response = "Error generating answer from LLM."
        
    final_output = f"{response}\n\nSources used:\n{sources}"
    return {"generation": final_output}

def decide_next_step(state: GraphState):
    """Conditional edge routing after document grading."""
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    retries = state.get("retries", 0)
    
    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE IRRELEVANT---")
        if retries < 2:
            print("---DECISION: RETRY / REWRITE QUERY---")
            return "rewrite_query"
        else:
            print("---DECISION: MAX RETRIES HIT, FALLING BACK TO WEB SEARCH---")
            return "web_search"
    else:
        print("---DECISION: CONTEXT IS RELEVANT, GENERATE ANSWER---")
        return "generate"

def check_hallucination(state: GraphState):
    """Determine if the generation is grounded in the document context."""
    print("---CHECK HALLUCINATION---")
    documents = state["documents"]
    generation = state["generation"]
    retries = state.get("retries", 0)
    
    if not documents:
        return "end"
        
    llm = get_llm(temperature=0)
    
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved documents. \n
    Give a binary score 'yes' or 'no'. 'yes' means the response is grounded and does not contain hallucinations. 'no' means the response contains information not supported by the context."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of documents: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    
    try:
        structured_llm_grader = llm.with_structured_output(HallucinationGrade)
        grader = grade_prompt | structured_llm_grader
        score = grader.invoke({"documents": "\n\n".join([d.page_content for d in documents]), "generation": generation})
        grade = score.binary_score
    except Exception as e:
        print(f"Error in structured hallucination check: {e}. Falling back to plain text parsing.")
        try:
            raw_grader = grade_prompt | llm
            raw_response = raw_grader.invoke({"documents": "\n\n".join([d.page_content for d in documents]), "generation": generation}).content
            grade = "no" if "no" in raw_response.lower() and "yes" not in raw_response.lower() else "yes"
        except Exception as ex:
            print(f"Fallback hallucination check also failed: {ex}")
            grade = "yes"  # Safely default to yes to avoid loop
        
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "end"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED (HALLUCINATION DETECTED)---")
        if retries < 2:
            print("---DECISION: RETRY QUERY REWRITING---")
            return "rewrite_query"
        else:
            print("---DECISION: MAX RETRIES HIT, END FLOW---")
            return "end"

# Build LangGraph workflow
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search", web_search)

# Set Graph Structure
workflow.set_entry_point("analyze_query")
workflow.add_edge("analyze_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Decide next step conditional edges
workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "web_search": "web_search"
    }
)

# If rewritten, re-retrieve
workflow.add_edge("rewrite_query", "retrieve")

# If web searched, proceed to generation
workflow.add_edge("web_search", "generate")

# Hallucination check conditional edge
workflow.add_conditional_edges(
    "generate",
    check_hallucination,
    {
        "end": END,
        "rewrite_query": "rewrite_query"
    }
)

app_graph = workflow.compile()

