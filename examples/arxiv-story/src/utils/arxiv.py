import io
import re
import aiohttp
from pypdf import PdfReader
from loguru import logger

async def get_arxiv_content(url: str, aiohttp_session: aiohttp.ClientSession) -> str:
    """Fetch and extract content from an arXiv paper URL."""
    # Extract the arXiv ID from the URL
    arxiv_id = None
    if "arxiv.org/abs/" in url:
        arxiv_id = url.split("arxiv.org/abs/")[-1].split()[0]
    elif "arxiv.org/pdf/" in url:
        arxiv_id = url.split("arxiv.org/pdf/")[-1].replace(".pdf", "").split()[0]
    
    if not arxiv_id:
        return "Could not extract arXiv ID from URL."

    # Construct the PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    # Set up headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        async with aiohttp_session.get(pdf_url, headers=headers, allow_redirects=True) as response:
            if response.status != 200:
                logger.error(f"Failed to download PDF: {response.status}")
                return f"Failed to download PDF. Status code: {response.status}"

            content = await response.read()
            
            # Check if we actually got a PDF
            if not content.startswith(b"%PDF"):
                logger.error("Response is not a PDF")
                return "The downloaded content is not a valid PDF."

            pdf_file = io.BytesIO(content)
            pdf_reader = PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = text.strip()

            return text

    except Exception as e:
        logger.error(f"Error downloading or processing PDF: {str(e)}")
        return f"Error processing the PDF: {str(e)}"

def truncate_content(content: str, model_name: str) -> str:
    """Truncate content to fit within model's token limit."""
    import tiktoken
    # Always use GPT-4's tokenizer (cl100k_base)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(content)

    max_tokens = 10000
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return content 