from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    YoutubeLoader,
)

# url = 'https://www.cnnbrasil.com.br'


def load_site(url):
    loader = WebBaseLoader(url)
    documents_list = loader.load()
    document = "\n\n".join([doc.page_content for doc in documents_list])

    return document


# video_id = '9DRn9RpR2vA'


def load_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False)
    documents_list = loader.load()
    document = "\n\n".join([doc.page_content for doc in documents_list])

    return document


# file_path = 'files/FuelConsumption_train.csv'


def load_csv(file_path):
    loader = CSVLoader(file_path)
    documents_list = loader.load()
    document = "\n\n".join([doc.page_content for doc in documents_list])

    return document


# file_path = 'files/02 - GridSearchCV.pdf'


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents_list = loader.load()
    document = "\n\n".join([doc.page_content for doc in documents_list])

    return document


# file_path = 'files/Anotações por Slide.txt'


def load_txt(file_path):
    loader = TextLoader(file_path)
    documents_list = loader.load()
    document = "\n\n".join([doc.page_content for doc in documents_list])

    return document


# print(load_txt(file_path))
