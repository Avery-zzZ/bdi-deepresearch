from ragflow_sdk import RAGFlow


def _check_healthy(rf: RAGFlow) -> None:
    try:
        rf.list_datasets(page_size=1)
    except Exception as err:
        msg = "Fail to connect to ragflow service"
        raise ConnectionError(msg) from err


def create_ragflow(api_key: str, base_url: str) -> RAGFlow:
    rf = RAGFlow(api_key, base_url)
    _check_healthy(rf)

    return rf
