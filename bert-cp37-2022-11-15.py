from transformers import BertJapaneseTokenizer, AutoModelForQuestionAnswering
import torch


model = AutoModelForQuestionAnswering.from_pretrained('transformers/output/')  
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 


file_path = "a.txt"
with open(file_path, "r", encoding="utf-8") as f:
	context = f.read()


#context  = 'インフルエンザの予約は2022年12月1日からです。COVID-19の予約は2022年3月25日からです。私は嶋田たかしです。嶋田病院で働いています。私は総務部に所属しています。いつも忙しいです。一方、Aさんは301号室に入院してます。また、Bさんは302号室に入院してます。'


def reply(context, question):

    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    output = model(**inputs)
    answer_start = torch.argmax(output.start_logits)  
    answer_end = torch.argmax(output.end_logits) + 1 
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    answer = answer.replace(' ', '')
    return answer


question = "COVID-19の予約はいつからですか？"
answer = reply(context, question)
print("question: " + question)
print("answer: " + answer)

question = "インフルエンザの予約はいつからですか？"
answer = reply(context, question)
print("question: " + question)
print("answer: " + answer)


question = "あなたの職場はどこですか？"
answer = reply(context, question)
print("question: " + question)
print("answer: " + answer)

question = "仕事はどうですか？"
answer = reply(context, question)
print("question: " + question)
print("answer: " + answer)

question = "あなたの勤務先は？"
answer = reply(context, question)
print("question: " + question)
print("answer: " + answer)

question = "Aさんはどこにいますか？"
answer = reply(context, question)
print("question: " + question)
print("answer: " + answer)

question = "何年度版の医科点数表の解釈ですか？"
answer = reply(context, question)
print("question: " + question)
print("answer: " + answer)