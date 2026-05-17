from opensearchpy import OpenSearch
from app.config import OPENSEARCH_URL


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=OPENSEARCH_URL.startswith("https"),
        verify_certs=False,
        ssl_show_warn=False,
        timeout=60,
    )
