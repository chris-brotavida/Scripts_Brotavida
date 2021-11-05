import re, wikipedia, unidecode
from bs4 import BeautifulSoup
from urllib.request import urlopen


def bs4_wiki_scraping(page):
    '''Scraping de texto de la wikipedia utilizando la librería bs4.
    
    Parameters
    ----------
    page : str
        Nombre de la página a revisar (OJO: No es la dirección completa).
        Por ejemplo, si la página es https://en.wikipedia.org/wiki/apple,
        lo que se debe ingresar es "apple".
        
    Returns
    -------
    text_out : str
        Texto obtenido a partir de la página.
    '''
    # Acondicionando el link de la página
    page = page.replace(' ', '_')
    page = unidecode.unidecode(page)
    
    # Especificando la dirección URL
    source = urlopen(f'https://en.wikipedia.org/wiki/{page}').read()

    # Preparando la sopa 
    soup = BeautifulSoup(source,'lxml')

    # Extrayendo el texto de los párrafos
    paragraphs = []
    for paragraph in soup.find_all('p'):
        paragraphs.append(str(paragraph.text))
    
    # Uniendo como texto plano
    text_out = ' '.join(paragraphs)
    
    # Dropear footnotes en brackets
    text_out = re.sub(r"\[.*?\]+", '', text_out)
    
    # Eliminar los non-breaking-spaces
    text_out = re.sub(r"\xa0", ' ', text_out)
    
    # Eliminar los code promotion
    text_out = re.sub(r"code: lat promoted to code: la ", '', text_out)

    # Replace '\n' (a new line) with '' and end the string at $1000.
    text_out = text_out.replace('\n', '')
    
    # Remover espacios extra
    text_out = re.sub(r'\s+', ' ', text_out)
    
    return text_out


def wiki_wiki_scraping(page):
    '''Scraping de texto de la wikipedia utilizando la librería bs4.
    
    Parameters
    ----------
    page : str
        Nombre de la página a revisar (OJO: No es la dirección completa).
        Por ejemplo, si la página es https://en.wikipedia.org/wiki/apple,
        lo que se debe ingresar es "apple".
        
    Returns
    -------
    text_out : str
        Texto obtenido a partir de la página.
    '''
    # Obteniendo el texto de la página de wikipedia
    text_out = wikipedia.WikipediaPage(title=page).content
    print(text_out)
    # Limpiando el texto
    text_out = re.sub(r'==.*?==+', '', text_out)
    text_out = text_out.replace('\n', '')
    
    return text_out


# Módulo de testeo
if __name__ == '__main__':
    text = bs4_wiki_scraping('apple')
    print(text)
