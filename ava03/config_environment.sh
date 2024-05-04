#!/bin/bash

# Função para verificar se um comando está disponível
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verifica se o Anaconda está instalado
if command_exists conda; then
    echo "Anaconda encontrado. Criando ambiente com environment.yml..."

    # Cria um ambiente usando o arquivo environment.yml
    conda env create -f environment.yml

    echo "Ambiente criado com sucesso."
else
    echo "Anaconda não encontrado. Usando venv para criar ambiente virtual..."

    # Verifica se o Python está instalado
    if command_exists python3; then
        # Define um nome para o ambiente virtual
        ENV_NAME="my_venv"
        
        # Cria um ambiente virtual usando venv
        python3 -m venv $ENV_NAME
        
        # Ativa o ambiente virtual
        source $ENV_NAME/bin/activate
        
        echo "Ambiente virtual criado. Instalando pacotes do requirements.txt..."
        
        # Instala os pacotes do arquivo requirements.txt
        pip install -r requirements.txt
        
        echo "Pacotes instalados com sucesso."
    else
        echo "Python3 não encontrado. Por favor, instale o Python3 para prosseguir."
    fi
fi