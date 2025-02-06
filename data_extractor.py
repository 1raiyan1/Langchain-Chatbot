from langchain_community.document_loaders import WebBaseLoader

# URL to extract data from
url = "https://brainlox.com/courses/category/technical"

# Initialize the LangChain WebBaseLoader
loader = WebBaseLoader(url)

# Load the web page content
documents = loader.load()

# Print extracted content
for doc in documents:
    print(doc.page_content[:1000])  # Print first 1000 characters for preview
