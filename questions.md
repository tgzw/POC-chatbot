# Chatbot questions

## 1. Diferencias entre 'completion' y 'chat' models

Un completion model es un LLM que toma un prompt y devuelve una continuación. Ha sido entrenado para predecir los siguientes tokens.

Por el contrario, un modelo chat-completion es capaz de devolver una respuesta que simule el otro lado de una conversación, cuando el prompt está formateado de la forma correcta indicándole que se trata de una conversación.

Modelos de chat-completion: GPT-3.5, GPT-4, Llama-2, Llama-3, Falcon, Alpaca, FLAN-T5, etc

## 2. ¿Cómo forzar a que el chatbot responda 'sí' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?

Hay distintas formas de hacerlo. Una forma directa sería, en el prompt inicial del runnable basado en LLM que hemos creado, indicarle en detalle el formato de salida que queremos. Así, podríamos escribir ahí directamente las especificaciones de cómo queremos que responda.
Haciendo uso de aproximaciones implementadas en Langchain, podemos usar los output_parsers, de forma que podríamos añadirle un StructuredOutputParser tanto a un Agent como a una Chain.

```python
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(
        name="source",
        description="source used to answer the user's question, should be a website.",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(temperature=0)
chain = prompt | model | output_parser
```


Si queremos que el formato de salida se adecue a una dataclass, por ejemplo, podemos definir esa clase como lo necesita Langchain, definiendo los atributos especificando, sobre todo su descripción. Podemos ver ejemplos de ello aquí: https://python.langchain.com/docs/use_cases/extraction/how_to/examples/


Otra opción, si no quisiéramos depender tanto de la infraestructura de Langchain, sería añadir una capa más a nuestro sistema, definiendo otro agente basado en LLM cuyo objetivo será tomar la respuesta del primer agente y transformarla a nuestro formato objetivo. Como antes, esto se conseguiría especificando en el prompt inicial los detalles de nuestro formato deseado.

Esta aproximación tiene como inconveniente que hacemos más llamadas a un LLM, aumentando tiempo de ejecución y posibles costes, pero tiene la ventaja de que usamos un agente especializado en una tarea. Esto permitiría que el prompt inicial del agente conversacional fuera menor y se le pidieran menos cosas (que dependiendo de la complejidad de las tareas que se le pidan, puede dar mejores resultados). 

Además, al estar usando un agente únicamente para el parseo, podríamos explorar opciones como hacer fine-tunning de algún modelo con un dataset de típicos inputs y outputs esperados, u otras técnicas como few-shot learning para especializar nuestro modelo en la tarea específica.

## 3. Ventajas e inconvenientes de RAG vs fine-tunning

Ante el objetivo de tener un modelo de lenguaje especializado en un ámbito de conocimiento, típicamente la aproximación era hacer fine-tunning. Pero actualmente observamos en el estado del arte que con LLMs y un corpus de conocimiento muy basto y no estructurado, empleando técnicas de RAG se logran tener mejores resultados con mucho menos trabajo y coste de computación, al ahorrarte el fine-tunning.

Algunos inconvenientes de usar fine-tunning son:
- La especialización que obtiene el LLM mediante el fine-tunning lo hace menos generalístico, de modo que puede deteriorar su desempeño en el resto de tasks como conversar, analizar, etc. Usando RAG el LLM mantiene sus capacidades del pre-training intactas, ya que solo le damos contexto adicional cuando queremos que nos responda a un input.

- La calidad del modelo final depende de la calidad y cantidad de los datos de entrenamiento, con lo costoso o difícil que puede ser generarlos. Con RAG, seguimos dependiendo de la fiabilidad de los datos, pero estos pueden estar desestructurados en formato texto, y no necesitamos una gran cantidad para obtener un buen desempeño.

- El conocimiento del modelo final no es dinámico. Si queremos actualizarlo o ampliarlo habría que entrenarlo de nuevo con los nuevos datos. Usando RAG, la adición de nuevos datos al contexto es tan rápida como indexar nuevos documentos.

Fine-tunning podría ser una opción buena a tomar cuando:
- Queremos un modelo muy especializado y no necesitamos hacer uso de las habilidades generales típicas de un LLM
- El conocimiento a usar para la especialización es estático y no contamos con que pueda ampliarse o modificarse.
- Estamos trabajando con un modelo de lenguaje mediano o pequeño (cuyo número de parámetros no llega a las decenas de billón) de modo que sus capacidades de conversación general son reducidas.

## 4. ¿Cómo evaluar el desempeño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG?


### RAG
Evaluar el desempeño de un RAG es complejo, debido a que el LLM conversacional que usemos por debajo en principio va a ser un sistema no determinista. Esto hace que, al contrario que con otro tipo de modelos de machine learning, no podemos hacer un dataset de evaluación y calcular accuracy, f1-score, etc. en base a las respuestas que coincidan con las de test.

Además, una respuesta correcta puede ser formulada de muchas formas, por lo que para evaluar cómo de correspondiente es una respuesta para una pregunta dada, necesitamos un sistema que comprenda la semántica de la pregunta y de la respuesta. Esto es contradictorio, porque si poseyéramos un sistema con una capacidad perfecta de hacer esa evaluación, bien podríamos usar dicho sistema para hacer la predicción de la respuesta en primer lugar.

Con esto en mente, hay una diversidad de parámetros que podemos tener en cuenta para evaluar un RAG:
- question (input)
- ground truth (el output "ideal" esperado)
- answer (output)
- retrieved context (los fragmentos de texto recuperados de la vector store)

Existe una diversidad de métricas que se han desarrollado para tratar de obtener métricas del desempeño de un sistema RAG. Uno de los frameworks que podrían usarse con este objetivo es Ragas. Algunas de estas métricas son:

#### Answer Correctness
Rango: 0-1  
Tiene en cuenta: input, output, ground truth  
Calcula el accuracy del output comparado con el ground truth  
Esta métrica combina 2 aspectos, la similaridad semántica y la similaridad factual.  

La corrección factual se determina calculando el f1-score de:
- Verdaderos positivos: hechos que están presentes en el output y en el ground truth
- Falsos positivos: hechos que están presentes en el output pero no en el ground truth
- Falsos negativos: hechos que están presentes en el ground truth pero no en el output

#### Answer Semantic Similarity
Rango: 0-1  
Tiene en cuenta: input, output, ground truth  
Aspira a calcular el parecido semántico del output comparado con el ground truth  
Para ello, vectoriza el ground truth y el output y calcula la distancia coseno entre los dos vectores.  

#### Answer Relevancy
Tiene en cuenta: input, output, retrieved context  
Determina la relevancia del output respecto al input  
Para ello, mide la distancia coseno del input original con un número de inputs artificiales generados a partir de la respuesta.  

#### Faithfulness
Tiene en cuenta: input, output, retrieved context  
Comprueba que el output esté alineado con el retrieved context y no haya inventado cosas  

#### Contextual Precision
Tiene en cuenta: input, output, retrieved context  
Comprueba si se ha recuperado contexto no relevante  

#### Contextual Relevancy
Tiene en cuenta: input, output, retrieved context  
Comprueba si el contexto recuperado es relevante y penaliza la información redundante  

#### Context Recall
Rango: 0-1  
Tiene en cuenta: retrieved context, ground truth  
Analiza cada frase del ground truth para determinar si puede ser atribuida al retrieved context o no.  
Para ello, se descompone el ground truth en frases individuales, y mediante un LLM se determina si cada una puede ser atribuida al retrieved context.  


### Q&A bot

Para evaluar el desempeño de un chatbot de Q&A, primero hemos de tener claro el scope del chatbot. 

Si queremos que tenga capacidades avanzadas de conversación general, querremos que las respuestas que dé frente a inputs de conversación casual tengan "sentido". Este sentido es subjetivo y será dependiente del tipo de público que vaya a tener el producto, qué uso se espera darle, etc., de forma que evaluar de forma adecuada su desempeño puede requerir apoyarse en human feedback. Haciendo uso de LLMs de trillones de parámetros, el ajuste del sistema para que su tono y actitud en conversación general sea el deseado es posible que pueda limitarse a ajustar el prompt inicial.

En esta línea, podríamos también usar métodos basados en LLMs para evaluar: toxicidad, sesgos, informalidad, etc. Con datasets de preguntas y respuestas obtenidas.

Si se trata de un chatbot con acceso a una diversidad de herramientas previamente definidas, que ante un input debe ser capaz de elegir la herramienta adecuada, o la combinación de herramientas (con inputs dependientes de los outputs de otras herramientas o no), puede ser útil evaluar que la elección de herramientas o recorrido entre herramientas sea el esperado ante un input concreto.

En última instancia, siempre vamos a querer evaluar si las respuestas que da ante ciertos inputs sean las esperadas. Para ello, podemos evaluar el Answer Correctness definido anteriormente con un dataset de preguntas, respuestas esperadas y respuestas obtenidas.

Además de todo lo comentado, también querremos tener en cuenta aspectos como: tiempo de respuesta, exigencia computacional, cantidad de tokens, etc.