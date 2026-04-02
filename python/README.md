# Leitor de Gabaritos (Versão Python Paralela)

Esta é a versão local, 100% Python, do sistema de processamento de gabaritos.
Diferente da versão web que possui interface visual, essa ferramenta foi criada para processamento em larga escala através do terminal.

### Benefícios

1. **Processamento Assíncrono e Local**: Escapa dos gargalos e timeouts da Vercel (limite de 60s). Você pode esvaziar mais de 100 fotos na pasta e deixá-lo corrigindo.
2. **Sistema de Rotacionamento**: Ao colocar múltiplas chaves (Google01, Google02...) ele usa e avança a chave no caso de dar rate limit `HTTP 429`.

### Passo a passo para rodar no MAC

1. Acesse esta pasta pelo terminal.
2. Ative o Virtual Env (já existente): `source venv/bin/activate`
   - *Se não existir, crie com `python3 -m venv venv`*
3. Instale/atualize as bibliotecas: `pip install -r requirements.txt`
4. Crie o arquivo de senhas, copiando o modelo `cp .env.example .env` e substitua as chaves falsas pelas suas reais.
5. Jogue as fotos reais das provas (formato JPG, PNG ou WEBP) na pasta que o robô monitora: `./imagens_provas/`
6. Rode o corretor: `python main.py`

Ao final do processamento o resultado fica disponível num arquivo .csv que você pode importar no Excel na pasta `./resultados/`
