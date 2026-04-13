import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import pandas as pd
import os, tempfile, subprocess, re
from datetime import datetime

st.set_page_config(page_title="AI Đọc File Thông Minh", layout="wide", page_icon="🤖")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage.user { background-color: #262730; border-radius: 12px; }
    .stChatMessage.assistant { background-color: #1e1f23; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
for key in ["documents", "file_summaries", "processed_files", "vectorstore", "last_doc_count", "messages", "chat_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["documents", "file_summaries", "messages", "chat_history"] else set() if key == "processed_files" else None if key == "vectorstore" else 0

# ================= PARSERS (Phần này là nguyên nhân lỗi) =================
def parse_excel_smart(path):
    df = pd.read_excel(path, header=None)
    employees = []
    current = None
    for i in range(len(df)):
        row = df.iloc[i].astype(str)
        text = " ".join(row)
        if "Tên nhân viên" in text:
            try:
                name = re.search(r"Tên nhân viên:\s*(.*?)\s*Phòng ban", text).group(1).strip()
                dept = re.search(r"Phòng ban:\s*(.*)", text).group(1).strip()
                current = {"name": name, "department": dept, "records": []}
                employees.append(current)
            except:
                continue
        elif current and any(":" in str(cell) for cell in row):
            current["records"].append([str(x) for x in row.tolist()])
    if employees:
        return employees
    else:
        return pd.read_excel(path, header=0) if len(df) > 0 else pd.DataFrame()

def excel_to_text(data, filename):
    if isinstance(data, list):
        text = f"📊 BÁO CÁO NHÂN VIÊN - {filename}\n\n"
        for emp in data:
            text += f"👤 {emp['name']} ({emp['department']})\n"
            text += f"- Số lần vi phạm: {len(emp['records'])}\n"
            for r in emp["records"]:
                text += f"  • {' | '.join(r)}\n"
            text += "\n"
    else:
        text = f"📊 EXCEL/CSV - {filename}\n\nDòng: {len(data)} | Cột: {len(data.columns)}\n\n"
        text += data.to_markdown(index=False)
    return text

def parse_docx_smart(path):
    import docx
    doc = docx.Document(path)
    sections = []
    current = {"title": "Nội dung chính", "content": []}
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading") and p.text.strip():
            if current["content"]:
                sections.append(current)
            current = {"title": p.text.strip(), "content": []}
        else:
            if p.text.strip():
                current["content"].append(p.text.strip())
    if current["content"]:
        sections.append(current)
    return sections

def docx_to_text(sections, filename):
    text = f"📄 WORD DOCUMENT - {filename}\n\n"
    for sec in sections:
        text += f"## {sec['title']}\n" + "\n".join(sec["content"]) + "\n\n"
    return text

def parse_pdf_smart(path):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    pages = loader.load()
    text = f"📕 PDF DOCUMENT - {os.path.basename(path)}\n\nTổng số trang: {len(pages)}\n\n"
    text += "\n\n".join([p.page_content for p in pages])
    return text

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🤖 AI Đọc File")
    st.caption("Hoàn toàn Offline")

    st.divider()
    def get_models():
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            return [line.split()[0] for line in result.stdout.splitlines() if line.strip() and not line.startswith("NAME")]
        except:
            return ["llama3.1", "qwen2.5"]

    model_name = st.selectbox("Model LLM", get_models())
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    st.divider()
    if st.button("🆕 Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        if st.session_state.messages:
            title = (st.session_state.messages[0]["content"][:35] + "...") if st.session_state.messages else "Cuộc chat mới"
            st.session_state.chat_history.append({
                "title": title,
                "messages": st.session_state.messages.copy(),
                "time": datetime.now().strftime("%d/%m %H:%M")
            })
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 📜 Lịch sử")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        if st.button(f"{chat['time']} - {chat['title']}", key=f"hist_{i}", use_container_width=True):
            st.session_state.messages = chat["messages"].copy()
            st.rerun()

    if st.button("🗑️ Xóa tất cả", type="secondary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ================= UPLOAD =================
st.markdown("### 📤 Upload file")

uploaded_files = st.file_uploader(
    "Excel, PDF, Word, CSV, TXT",
    type=["xlsx","xls","csv","pdf","docx","txt"],
    accept_multiple_files=True
)

if uploaded_files:
    new_added = False
    for file in uploaded_files:
        if file.name in st.session_state.processed_files:
            continue
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            if file.name.endswith((".xlsx", ".xls", ".csv")):
                data = parse_excel_smart(tmp_path)
                text = excel_to_text(data, file.name)
                summary = f"📊 {file.name}"
                docs_to_add = [Document(page_content=text, metadata={"source": file.name})]

            elif file.name.endswith(".docx"):
                sections = parse_docx_smart(tmp_path)
                text = docx_to_text(sections, file.name)
                summary = f"📄 {file.name}"
                docs_to_add = [Document(page_content=text, metadata={"source": file.name})]

            elif file.name.endswith(".pdf"):
                text = parse_pdf_smart(tmp_path)
                summary = f"📕 {file.name}"
                docs_to_add = [Document(page_content=text, metadata={"source": file.name})]

            else:  # txt
                with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()[:15000]
                text = f"📄 TEXT - {file.name}\n\n{content}"
                summary = f"📄 {file.name}"
                docs_to_add = [Document(page_content=text, metadata={"source": file.name})]

            st.session_state.documents.extend(docs_to_add)
            st.session_state.file_summaries.append(summary)
            st.session_state.processed_files.add(file.name)
            new_added = True

        except Exception as e:
            st.error(f"Lỗi khi đọc **{file.name}**: {e}")
        finally:
            try: os.unlink(tmp_path)
            except: pass

    if new_added:
        st.success("✅ File đã được tải lên!")
        st.session_state.vectorstore = None   # Buộc tạo lại vectorstore
        st.rerun()

# ================= VECTORSTORE & CHAT =================
if st.session_state.documents and st.session_state.vectorstore is None:
    with st.spinner("Đang tạo chỉ mục tìm kiếm..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        splits = splitter.split_documents(st.session_state.documents)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vectorstore = Chroma.from_documents(splits, embeddings)
    st.toast("✅ Sẵn sàng hỏi đáp!", icon="🚀")

st.markdown("### 💬 Hỏi AI về dữ liệu")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Hỏi về file đã upload..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.vectorstore:
        response = "⚠️ Vui lòng upload file trước khi hỏi."
    else:
        with st.spinner("Đang phân tích..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})
            docs = retriever.invoke(prompt)
            context = "\n\n".join([f"--- File: {d.metadata['source']} ---\n{d.page_content[:1800]}" for d in docs])

            llm = ChatOllama(model=model_name, temperature=temperature)
            system_prompt = "Bạn là trợ lý phân tích tài liệu thông minh. Trả lời bằng tiếng Việt, rõ ràng, chính xác và trích dẫn file nếu cần."

            chain = (
                ChatPromptTemplate.from_template(f"{system_prompt}\n\nContext:\n{context}\n\nCâu hỏi: {{question}}")
                | llm
                | StrOutputParser()
            )
            response = chain.invoke({"question": prompt})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

if not st.session_state.file_summaries:
    st.info("📤 Hãy upload file để bắt đầu sử dụng.", icon="📌")