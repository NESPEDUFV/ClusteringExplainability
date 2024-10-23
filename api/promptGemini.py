# %pip install -q -U google-generativeai
# %pip install python-dotenv

#import google.generativeai as gemini
#from dotenv import load_dotenv
import os

class PromptGemini:
    def __init__(self, cluster):
        self.cluster = cluster
        self.semantic = ""

    async def generate(self, isSemantic=False):
        try:
            file_content = self.read_txt_file('api\semantica_diabetes.txt')
            self.semantic = file_content
        except ValueError as e:
            print(e)
            return
            
        #load_dotenv()
        #API_KEY = os.getenv("API_KEY")
        #gemini.configure(api_key=API_KEY)

        # for m in gemini.list_models():
        #     if 'generateContent' in m.supported_generation_methods:
        #         print(m.name)

        #model = gemini.GenerativeModel("gemini-1.5-flash-latest")

        prompt = """
        Você é um AI especialista em interpretar clusters de dados. Sua tarefa é transformar a descrição de clusters 
        gerados por um algoritmo em narrativas claras e acessíveis para o público leigo. Cada cluster é definido por 
        um conjunto de características formais. Para gerar a descrição, siga os passos abaixo:
        1. Identifique as Características Principais: Extraia as características mais relevantes dos clusters fornecidos.
        2. Contextualize e Simplifique: Explique o que essas características significam no contexto do grupo descrito. 
        Detalhe as informações e use uma linguagem simples e direta para garantir que qualquer pessoa possa entender.
        3. Características Marcantes: Destaque uma característica única que diferencia o cluster dos demais. Se não houver 
        tal característica, informe que o cluster não possui uma característica marcante.
        4. Comparação das Características Marcantes: Compare as características marcantes de cada cluster. Elabore sobre 
        como essas características se diferenciam entre os clusters.
        5. Métricas do Cluster: Considere as seguintes métricas para cada cluster:
        - **Coverage**: Mede a proporção de pontos de dados que pertencem a um cluster específico e que são descritos pela 
        explicação. Uma alta cobertura indica que a explicação é válida para a maioria dos pontos do cluster.
        - **Separation Error**: Mede a proporção de pontos que a explicação considera válidos, mas que na verdade 
        pertencem a outros clusters. Um baixo erro de separação é desejável, pois indica que a explicação é específica 
        para o cluster em questão e não se aplica a pontos de outros clusters.
        - **Conciseness**: Refere-se à simplicidade e brevidade da explicação. Uma explicação concisa é mais fácil de 
        entender e interpretar. A concisão é medida pelo número de predicados que compõem a explicação; quanto menor 
        o número de predicados, maior a concisão.
        6. Avalie para cada cluster: Forneça uma avaliação qualitativa de cada cluster com base nas métricas fornecidas. 
        Formato de Saída: Apresente suas descrições em parágrafos claros, assegurando que sejam compreensíveis e 
        informativas para um público leigo.
        **Exemplo de Entrada:**
        "<característica, 80-between, intervalo>"
        Neste exemplo, "80-between" significa que 80% dos dados do cluster estão dentro do intervalo especificado. Isso 
        indica que a maior parte dos pontos do cluster está contida entre os limites descritos, fornecendo uma ideia 
        sobre a distribuição dos dados em relação àquela característica.
        Outro exemplo de entrada: "<sex, contains, 1.0>" (onde "contains" indica que a característica 'sexo' do paciente 
        contém um valor específico, neste caso, 1.0)
        **Observação:** Semânticas podem ser enviadas junto com os clusters e devem ser usadas para compreender 
        o significado de cada característica, caso estejam presentes. Ela virá com o título "Semântica" e será seguida
        de uma lista de características e seus significados. 
        Exemplo: 
        Semânticas
        age: Idade do paciente.
        """

        semantica = str(self.semantic)
        cluster = str(self.cluster)

        pergunta = semantica + "\n\n" + cluster
        
        print("\n\n")
        print(pergunta)
        
        print("\n")
        print("Gerando descrição do cluster...")
        print("\n")

        #chat = model.start_chat(history=[])

        #response = chat.send_message(prompt)
        #response = chat.send_message(pergunta)
        
        #print(response.text)
        
        #self.save_response_to_file(response.text, directory="out", filename="output.md")
       
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
        
    def read_txt_file(self, file_path):
        """
        Lê o conteúdo de um arquivo .txt e retorna como uma string.

        :param file_path: Caminho para o arquivo .txt
        :return: Conteúdo do arquivo como uma string
        :raises ValueError: Se o arquivo não for encontrado ou ocorrer um erro ao ler o arquivo.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            raise ValueError(f"O arquivo '{file_path}' não foi encontrado.")
        except Exception as e:
            raise ValueError(f"Ocorreu um erro ao ler o arquivo: {str(e)}")


