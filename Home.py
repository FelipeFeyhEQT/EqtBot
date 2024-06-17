import time
import streamlit as st
import streamlit.components.v1 as components
from utils import cria_chain_conversa, PASTA_ARQUIVOS

def sidebar():
    uploaded_pdfs = st.file_uploader(
        'Adicione seus arquivos', 
        type=['.pdf'], 
        accept_multiple_files=True
        )
    if uploaded_pdfs:
        for arquivo in PASTA_ARQUIVOS.glob('*.pdf'):
            arquivo.unlink()
        for pdf in uploaded_pdfs:
            with open(PASTA_ARQUIVOS / pdf.name, 'wb') as f:
                f.write(pdf.read())
    
    label_botao = 'Inicializar ChatBot'
    if 'chain' in st.session_state:
        label_botao = 'Atualizar ChatBot'
    if st.button(label_botao, use_container_width=True):
        if len(list(PASTA_ARQUIVOS.glob('*.pdf'))) == 0:
            st.error('Adicione arquivos .pdf para inicializar o chatbot')
        else:
            st.success('Inicializando o ChatBot...')
            cria_chain_conversa()
            st.rerun()

    st.sidebar.image('pages/logo-eqtlab-nobg.png', use_column_width=True)

def chat_window():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #001f3f; /* Cor de fundo azul marinho */
            color: white; /* Cor branca para o texto */
        }
        .footer {
            position: fixed;
            left: 50%;
            bottom: 10px;
            transform: translateX(-50%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.header('🤖 Bem-vindo ao Chat com Docs do EQTLAB', divider=True)

    if 'chain' not in st.session_state:
        st.error('Faça o upload de PDFs para começar!')
        st.stop()
    
    chain = st.session_state['chain']
    memory = chain.memory

    mensagens = memory.load_memory_variables({})['chat_history']

    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    nova_mensagem = st.chat_input('Converse com seus documentos...')
    if nova_mensagem:
        chat = container.chat_message('human')
        chat.markdown(nova_mensagem)
        chat = container.chat_message('ai')
        chat.markdown('Gerando resposta')

        resposta = chain.invoke({'question': nova_mensagem})
        st.session_state['ultima_resposta'] = resposta
        st.experimental_rerun()

    st.markdown(
        """
        <div class="footer">
            <img src="pages/logo.jpg" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    with st.sidebar:
        sidebar()
    chat_window()

if __name__ == '__main__':
    main()
