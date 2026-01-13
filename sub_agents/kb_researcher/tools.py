import json

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from ragflow_sdk import Chunk, DataSet, Document, RAGFlow


def _get_dataset_name(runtime: ToolRuntime) -> str:
    return runtime.context.kb_ragflow_dataset_name


def _get_dataset(ragflow: RAGFlow, runtime: ToolRuntime) -> DataSet:
    ds_name = _get_dataset_name(runtime)
    # TODO: should check dataset existency first somewhere.
    return ragflow.get_dataset(ds_name)


def _get_dataset_id(ragflow: RAGFlow, runtime: ToolRuntime) -> str:
    if ds_id := getattr(runtime.context, "kb_ragflow_dataset_id", None):
        return ds_id

    ds = _get_dataset(ragflow, runtime)
    runtime.context.kb_ragflow_dataset_id = ds.id
    return ds.id


def _get_docs(ds: DataSet, page: int, page_size: int, keywords: str) -> list[Document]:
    if not page_size or ds.document_count < (page - 1) * page_size:
        return []

    return ds.list_documents(page=page, page_size=page_size, keywords=keywords)


def _get_dataset_dscr(ds: DataSet) -> str:
    dscr = f"\n介绍: {ds.description}" if ds.description else ""
    return f"名称: {ds.name}\n文档总数: {ds.document_count}{dscr}"


def _get_docs_dscr(docs: list[Document]) -> str:
    doc_lines = [f"名称: {d.name}\tID: {d.id}" for d in docs]

    return "\n".join(doc_lines)


def _get_dataset_by_id(rf: RAGFlow, ds_id: str) -> DataSet:
    _list = rf.list_datasets(id=ds_id)
    if len(_list) > 0:
        return _list[0]
    msg = f"Dataset {ds_id} not found"
    raise ValueError(msg)


def _get_chunk_doc_name(chunk: Chunk, ds: DataSet) -> str:
    return ds.list_documents(chunk.document_id)[0].name


def get_chunk_doc_name(rf: RAGFlow, chunk: Chunk) -> str:
    ds = _get_dataset_by_id(rf, chunk.dataset_id)
    return _get_chunk_doc_name(chunk, ds)


def _get_chunk_return(c: Chunk) -> dict:
    return {
        "content": c.content,
        "position_in_doc": c.positions[0][1] if c.positions else None,
    }


def _get_chunks_return(rf: RAGFlow, chunks: list[Chunk]) -> list[dict]:
    chunks_by_doc = {}
    for c in chunks:
        if not chunks_by_doc.get(c.document_id, None):
            chunks_by_doc[c.document_id] = {
                "doc_id": c.document_id,
                "doc_name": get_chunk_doc_name(rf, c),
                "retrieved_chunks": [
                    _get_chunk_return(c),
                ],
            }
            continue
        chunks_by_doc[c.document_id]["retrieved_chunks"].append(
            _get_chunk_return(c),
        )
    return list(chunks_by_doc.values())


def _create_tools(ragflow: RAGFlow) -> dict:
    ls_docs_tool_description = """列出知识库中的文档"""
    ls_docs_tool_name = "kb_ls_docs"

    @tool(ls_docs_tool_name, description=ls_docs_tool_description)
    def ls_docs(runtime: ToolRuntime, page: int = 1, page_size: int = 20) -> str:
        ds = _get_dataset(ragflow, runtime)
        docs: list[Document] = _get_docs(ds, page, page_size)
        if not docs:
            return f"无结果（文档总数: {ds.document_count}）"
        return f"本页有{len(docs)}个文档（文档总数: {ds.document_count}）：\n{_get_docs_dscr(docs)}"

    search_chunk_tool_description = """搜索知识库中的文本块（稠密检索）
Args:
    - query(str) :检索语句
    - max_return(int) :最大返回数量(默认值：10)
"""
    search_chunk_tool_name = "kb_search_chunk"

    @tool(search_chunk_tool_name, description=search_chunk_tool_description)
    def search(runtime: ToolRuntime, query: str, max_return: int = 10) -> dict:
        ds_id = _get_dataset_id(ragflow, runtime)
        chunks: list[Chunk] = ragflow.retrieve(
            [ds_id],
            question=query,
            page_size=max_return,
            similarity_threshold=0.3,
            vector_similarity_weight=0.4,
        )

        total_search = runtime.state.get("kb_total_search", 0) + 1
        empty_search = runtime.state.get("kb_empty_search", 0)
        if not chunks:
            empty_search += 1
            if empty_search >= 3 and empty_search == total_search:
                return "<system>/command 系统已确认知识库中无相关信息，立即调用`research_done`工具汇报未找到有用信息</system>"

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(
                            {"chunks_by_doc": _get_chunks_return(ragflow, chunks)},
                        ),
                        tool_call_id=runtime.tool_call_id,
                    ),
                ],
                "kb_total_search": total_search,
                "kb_empty_search": empty_search,
            },
        )

    return {ls_docs_tool_name: ls_docs, search_chunk_tool_name: search}
