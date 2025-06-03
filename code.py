from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, T5ForConditionalGeneration,
    pipeline, AutoModelForQuestionAnswering
)
import torch
import re
import random
import io
import logging

# Setup
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models
sum_model_name = "t5-small"
sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
sum_model = T5ForConditionalGeneration.from_pretrained(sum_model_name)

try:
    question_gen_pipeline = pipeline("e2e-qg")
except Exception:
    question_gen_pipeline = None

qa_model_name = "distilbert-base-cased-distilled-squad"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sum_model.to(device)
qa_model.to(device)


# Utility functions
def clean_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def summarize(text: str) -> str:
    try:
        input_text = "summarize: " + clean_text(text)
        input_tokens = sum_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        input_word_len = len(text.split())
        max_length = max(50, int(input_word_len * 0.4))
        min_length = max(30, int(max_length * 0.9))

        summary_ids = sum_model.generate(
            input_tokens,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.85,
            num_return_sequences=1,
            early_stopping=True
        )
        return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return "Could not generate summary."

def extract_keywords(text, num_keywords=20):
    stopwords = set([
        "the", "and", "is", "in", "to", "of", "a", "for", "on",
        "with", "as", "by", "an", "be", "at", "from", "that",
        "this", "it", "are", "was", "or", "but", "if", "or"
    ])
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        if word not in stopwords and len(word) > 2:
            freq[word] = freq.get(word, 0) + 1
    sorted_keywords = sorted(freq, key=freq.get, reverse=True)
    return sorted_keywords[:num_keywords]

def generate_quiz(text: str, min_questions=5):
    if question_gen_pipeline:
        try:
            generated = question_gen_pipeline(text)
            questions = []
            for item in generated[:min_questions]:
                question_text = item['question']
                answer_text = item['answer']
                options = [answer_text] + random.sample([k for k in extract_keywords(text) if k != answer_text], k=3)
                random.shuffle(options)
                questions.append({"question": question_text, "options": options, "answer": answer_text})
            return questions
        except Exception:
            pass

    sentences = [s.strip() for s in re.split(r'[.?!]', text) if len(s.split()) > 5]
    keywords = extract_keywords(text)
    questions = []
    used_sentences = set()
    random.shuffle(sentences)

    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence.lower() and sentence not in used_sentences:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                question_text = pattern.sub('______', sentence, count=1)
                options = [keyword] + random.sample([k for k in keywords if k != keyword], k=3)
                random.shuffle(options)
                questions.append({"question": f"Fill in the blank: {question_text}", "options": options, "answer": keyword})
                used_sentences.add(sentence)
                if len(questions) >= min_questions:
                    return questions
    return questions[:min_questions]

def quiz_to_text(questions):
    output = io.StringIO()
    for i, q in enumerate(questions, 1):
        output.write(f"Q{i}: {q['question']}\n")
        for j, opt in enumerate(q['options']):
            output.write(f"   {chr(65+j)}. {opt}\n")
        output.write("\n")
    return output.getvalue()

def answers_to_text(questions):
    output = io.StringIO()
    for i, q in enumerate(questions, 1):
        idx = q['options'].index(q['answer'])
        output.write(f"Q{i}: {chr(65+idx)}. {q['answer']}\n")
    return output.getvalue()

def answer_question(context: str, question: str) -> str:
    try:
        inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.convert_tokens_to_string(
            qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        )
        return answer if answer.strip() else "No answer found."
    except Exception as e:
        logger.warning(f"QA failed: {e}")
        return "Failed to answer question."

# API Schemas
class QARequest(BaseModel):
    context: str
    question: str

class TextRequest(BaseModel):
    text: str

# Endpoints
@app.post("/summarize", response_class=PlainTextResponse)
def summarize_text(req: TextRequest):
    return summarize(req.text)

@app.post("/generate-quiz", response_class=PlainTextResponse)
def get_quiz(req: TextRequest):
    quiz = generate_quiz(req.text)
    return quiz_to_text(quiz)

@app.post("/quiz-answers", response_class=PlainTextResponse)
def get_quiz_answers(req: TextRequest):
    quiz = generate_quiz(req.text)
    return answers_to_text(quiz)

@app.post("/qa", response_class=PlainTextResponse)
def get_answer(req: QARequest):
    return answer_question(req.context, req.question)

@app.post("/upload-file", response_class=PlainTextResponse)
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")
    return summarize(text)
def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()



# To run the FastAPI app, use the command:
# uvicorn code:app --reload
# This will start the server and you can access the API at http://localhost:8000/docs

