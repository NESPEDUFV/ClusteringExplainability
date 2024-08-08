

import google.generativeai as genai
from dotenv import load_dotenv
import os

class PromptGemini:
    def __init__(self, cluster, semantic):
        self.cluster = cluster
        self.semantic = semantic

    def generate(self):
        load_dotenv()
        API_KEY = os.getenv("API_KEY")
        genai.configure(api_key=API_KEY)

        #for m in genai.list_models():
            #if 'generateContent' in m.supported_generation_methods:
                #print(m.name)

        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        prompt = """
        Você é um AI especialista em interpretar clusters de dados. Sua tarefa é transformar a descrição de clusters gerados por um algoritmo em narrativas claras, detalhadas e compreensíveis para o público leigo. Cada cluster é definido por um conjunto de características formais, e você deve elaborar uma descrição rica e acessível, fornecendo contexto e explicações quando necessário.
        Antes de seguir o passo a passo o algoritmo pode gerar também um arquivo de semântica, essa semântica indica o que é cada característica. Exemplo:
        Arquivo de semânticas : A1 : <característica>, isso significa que essa característica é representada por A1. Primeiramente verifique se junto com os clusters foi enviado o arquivo chamado Semânticas, depois realize o passo a passo.
        Você deve seguir os seguintes passos:
        1. Identifique as Características Principais: Extraia as características mais relevantes dos clusters fornecidos.
        2. Contextualize e Simplifique: Explique o que essas características significam no contexto do grupo descrito. Detalhe as informações obtidas e as contextualizes. Agora simplifique, use uma linguagem simples e direta para garantir que qualquer pessoa possa entender a descrição.
        3. Característica marcante: Destaque uma característica caso existir que faz aquele cluster ser único, que o diferencie dos demais, é preferível que nenhum outro cluster tenha essa características, caso não encontre informe que o clusters não tem característica marcante.
        4. Comparação das características marcantes: Faça comparações das características marcantes de cada cluster.
        """

        semantica = str(self.semantic)
        cluster = str(self.cluster)

        pergunta = semantica + cluster
        
        print(pergunta)

        chat = model.start_chat(history=[])

        response = chat.send_message(prompt)
        response = chat.send_message(pergunta)
        print(response.text)
        
    def format_description(cluster_number, description_user):
        """
        Formata uma lista de descrições de um cluster em uma string organizada.

        Parâmetros:
        - cluster_number: Número do cluster.
        - description_user: Lista de strings contendo as descrições do cluster.

        Retorno:
        - Uma string formatada contendo as descrições do cluster.
        """
        formatted_description = f"cluster {cluster_number}\n" + "\n".join(description_user)
        return f'Description user:\n"""\n{formatted_description}\n"""'