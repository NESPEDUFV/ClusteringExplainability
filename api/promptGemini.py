# %pip install -q -U google-generativeai
# %pip install python-dotenv

import google.generativeai as gemini
from dotenv import load_dotenv
import os

class PromptGemini:
    def __init__(self, cluster, semantic):
        self.cluster = cluster
        self.semantic = semantic

    async def generate(self):
        load_dotenv()
        API_KEY = os.getenv("API_KEY")
        gemini.configure(api_key=API_KEY)

        #for m in gemini.list_models():
            #if 'generateContent' in m.supported_generation_methods:
                #print(m.name)

        model = gemini.GenerativeModel("gemini-1.5-pro-latest")

        prompt = """
        Você é um AI especialista em interpretar clusters de dados. Sua tarefa é transformar a descrição de clusters gerados 
        por um algoritmo em narrativas claras, detalhadas e compreensíveis para o público leigo. 
        Cada cluster é definido por um conjunto de características formais, e você deve elaborar 
        uma descrição rica e acessível, fornecendo contexto e explicações quando necessário.
        Antes de seguir o passo a passo o algoritmo pode gerar os cluster no formato 
        <característica, 80-between, intervalo>. 80-between mostra que 80% dos dados estão entre certo intervalo indicado.
        Para gerar a Descrição você deve seguir os seguintes passos:
        1. Identifique as Características Principais: Extraia as características mais relevantes dos clusters fornecidos.
        2. Contextualize e Simplifique: Explique o que essas características significam no contexto do grupo descrito. Detalhe as informações obtidas e as contextualizes. Agora simplifique, use uma linguagem simples e direta para garantir que qualquer pessoa possa entender a descrição.
        3. Característica marcante: Destaque uma característica caso existir que faz aquele cluster ser único, que o diferencie dos demais, é preferível que nenhum outro cluster tenha essa características, caso não encontre informe que o clusters não tem característica marcante.
        4. Comparação das características marcantes: Faça comparações das características marcantes de cada cluster.
        """

        semantica = str(self.semantic)
        cluster = str(self.cluster)

        #pergunta = semantica + cluster
        pergunta = cluster
        
        print("\n")
        print("Gerando descrição do cluster...")
        print("\n")

        chat = model.start_chat(history=[])

        response = chat.send_message(prompt)
        response = chat.send_message(pergunta)
        
        self.save_response_to_file(response.text, directory="out", filename="output.md")
       
    @staticmethod 
    def format_cluster_descriptions(cluster_descriptions):
        formatted_output = ""
        for i, description in enumerate(cluster_descriptions):
            formatted_output += f"cluster {i}:\n"
            for line in description:
                formatted_output += f"{line}\n"
            formatted_output += "\n"
        return formatted_output.strip()
    
    def save_response_to_file(self, response, directory="../out", filename="output.md"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filepath = os.path.join(directory, filename)
        
        with open(filepath, "w") as file:
            file.write(response)
        
        print(f"Response saved to {filepath}")