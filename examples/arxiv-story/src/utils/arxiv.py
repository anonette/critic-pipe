import io
import aiohttp
from pypdf import PdfReader

async def get_arxiv_content(url: str, aiohttp_session: aiohttp.ClientSession) -> str:
    """Fetch and extract content from an arXiv paper URL."""
    if "/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    if not url.endswith(".pdf"):
        url += ".pdf"

    async with aiohttp_session.get(url) as response:
        if response.status != 200:
            return "Failed to download arXiv PDF."

        content = await response.read()
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def truncate_content(content: str, model_name: str) -> str:
    """Truncate content to fit within model's token limit."""
    import tiktoken
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(content)

    max_tokens = 10000
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return content 