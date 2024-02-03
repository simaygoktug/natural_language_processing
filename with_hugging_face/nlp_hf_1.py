#Bugünkü amaç modeli deploy ederek üretim ortamına aktarmak ve kullanıcıların hizmetine sunmak.
#Gradio ile kolay bir şekilde bunu yapacağım.

#Fine Tune edilmiş Roberta LLM'i kullanacağım.
#Başka model kullanmak için Hugging Face üzerinden --> Models --> NLP Models --> Most Downloaded --> Örneğin BERT tabanlı başka bir modelin yolunu kopyalayarak aşağıdaki path'e onu da yazabiliriz.

import transformers 
from transformers import pipeline

ner_pipeline = pipeline("ner", model="Tirendaz/roberta-base-NER")
text = "I am Tim and I work at FaceBook."
ner_pipeline(text) 
#Subword tekniği ile eğitilmiş modelde örneğin Trendyol kelimesini "Face" ve "Book" olarak bölmemek için:
ner_pipeline(text, aggregation_strategy="simple")

def ner(text):
  output = ner_pipeline(text, aggregation_strategy="simple")
  return {"text": text, "entities": output}
ner(text)

import gradio as gr

examples = [
    "My name is Tim and I live in California",
    "Ich arbeite bei Google in Berlin",
    "Ali, Ankara'li mi?"
]

demo = gr.Interface(
    ner,
    gr.Textbox(placeholder="Enter a sentence here..."),
    gr.HighlightedText(),
    examples = examples
)

demo.launch(share = True)
