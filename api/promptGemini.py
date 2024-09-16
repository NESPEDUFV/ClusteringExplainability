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

        # for m in gemini.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(m.name)

        model = gemini.GenerativeModel("gemini-1.5-flash-latest")

        prompt = """
        Você é um AI especialista em interpretar clusters de dados. Sua tarefa é transformar a descrição de clusters 
        gerados por um algoritmo em narrativas claras e acessíveis para o público leigo. Cada cluster é definido por 
        um conjunto de características formais. Para gerar a descrição, siga os passos abaixo:
        1. Identifique as Características Principais: Extraia as características mais relevantes dos clusters fornecidos.
        2. Contextualize e Simplifique: Explique o que essas características significam no contexto do grupo descrito. Detalhe as informações e use uma linguagem simples e direta para garantir que qualquer pessoa possa entender.
        3. Características Marcantes: Destaque uma característica única que diferencia o cluster dos demais. Se não houver tal característica, informe que o cluster não possui uma característica marcante.
        4. Comparação das Características Marcantes: Compare as características marcantes de cada cluster. Elabore sobre como essas características se diferenciam entre os clusters.
        5. Métricas do Cluster: Considere as seguintes métricas para cada cluster:
            Coverage: Mede a proporção de pontos de dados que pertencem a um cluster específico e que são descritos pela explicação. Uma alta cobertura indica que a explicação é válida para a maioria dos pontos do cluster.
            Separation Error: Mede a proporção de pontos que a explicação considera válidos, mas que na verdade pertencem a outros clusters. Um baixo erro de separação é desejável, pois indica que a explicação é específica para o cluster em questão e não se aplica a pontos de outros clusters.
            Conciseness: Refere-se à simplicidade e brevidade da explicação. Uma explicação concisa é mais fácil de entender e interpretar. A concisão é medida pelo número de predicados que compõem a explicação; quanto menor o número de predicados, maior a concisão.
        6. Avalie para cada cluster: Forneça uma avaliação qualitativa de cada cluster com base nas métricas fornecidas. Considere a eficácia da descrição, a clareza da explicação e a relevância das características destacadas.    
        Formato de Saída: Apresente suas descrições em parágrafos claros, assegurando que sejam compreensíveis e informativas para um público leigo.
        Exemplo de Entrada: "<característica, 80-between, intervalo>
        """

        semantica = str(self.semantic)
        cluster = str(self.cluster)

        pergunta = semantica + cluster
        pergunta = cluster
        
        print("\n")
        print("Gerando descrição do cluster...")
        print("\n")

        chat = model.start_chat(history=[])

        response = chat.send_message(prompt)
        response = chat.send_message(pergunta)
        
        print(response.text)
        
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