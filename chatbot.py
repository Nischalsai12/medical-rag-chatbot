import time
import functools
import torch
import psutil
import json
import faiss  #Facebook AI Similarity Search
import numpy as np
from typing import Callable, List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from evaluate import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gc
import os
import logging
from flask import Flask, request, render_template, jsonify

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Flask App Setup
app = Flask(__name__)

# Utility Functions
def time_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def evaluate_response(generated_response: str, ground_truth: str = None) -> Dict[str, Any]:
    """Evaluating the generated response with the BLEU, ROUGE, and word overlap."""
    results = {
        "length": len(generated_response),
        "word_count": len(generated_response.split())
    }
    
    if ground_truth:
        bleu = load("bleu") #Evaluation
        rouge = load("rouge") #Recall-Oriented Understudy for Gisting Evaluation
        bleu_score = bleu.compute(predictions=[generated_response], references=[[ground_truth]])
        rouge_score = rouge.compute(predictions=[generated_response], references=[ground_truth])
        generated_words = set(generated_response.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        overlap = len(generated_words.intersection(ground_truth_words))
        results.update({
            "bleu": bleu_score["bleu"],
            "rouge": rouge_score["rougeL"],
            "word_overlap": overlap / len(ground_truth_words) if ground_truth_words else 0
        })
    else:
        results.update({
            "bleu":None,
            "rouge": None,
            "word_overlap": None
        })
    
    return results

def baseline_keyword_search(query: str, faqs: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    """Keyword-based search baseline using TF-IDF."""
    questions = [faq['question'] for faq in faqs]
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [faqs[i] for i in top_k_indices]

def memory_statics():
    """Format memory usage statistics."""
    system_stats = {
        "RAM": f"{psutil.virtual_memory().used / (1024 * 3):.1f}GB / {psutil.virtual_memory().total / (1024 * 3):.1f}GB",
        "RAM Usage": f"{psutil.virtual_memory().percent}%"
    }
    
    if torch.cuda.is_available():
        gpu_stats = {}
        for i in range(torch.cuda.device_count()):
            gpu_stats[f"GPU {i}"] = f"{torch.cuda.get_device_name(i)}"
            gpu_stats[f"GPU {i} Memory"] = f"{torch.cuda.memory_allocated(i) / (1024 * 3):.1f}GB / {torch.cuda.get_device_properties(i).total_memory / (1024 * 3):.1f}GB"
        system_stats.update(gpu_stats)
    
    return system_stats

def check_memory():
    """Check available memory and warn if low."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    logger.debug(f"Available memory: {available_gb:.1f}GB")
    if available_gb < 2:
        logger.warning("Low memory (<2GB available). Increase paging file or free up RAM.")

# Embedding Class
class MedicalFAQEmbedder:
    def __init__(self, model_name: str =os.getenv("EMBEDDING_MODEL","all-MiniLM-L12-v2")):
        """Initialize the FAQ embedder with a sentence transformer model."""
        check_memory()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Embedding model using device: {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.debug("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {str(e)}")
            logger.info("Falling back to all-MiniLM-L6-v2...")
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                logger.debug("Fallback embedding model loaded successfully")
            except Exception as fallback_e:
                logger.error(f"Fallback embedding model error: {str(fallback_e)}")
                raise RuntimeError("Failed to load both primary and fallback embedding models")
        self.index = None
        self.faqs = None
        self.embeddings = None
    
    def create_embeddings(self, faqs: List[Dict[str, Any]], batch_size: int = 8) -> None:
        """Create embeddings for all FAQs and build FAISS index."""
        self.faqs = faqs
        check_memory()
        logger.debug(f"Creating embeddings for {len(faqs)} FAQs in batches of {batch_size}...")
        
        questions = [faq['question'] for faq in faqs]
        all_embeddings = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
            batch_embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
            gc.collect()
        
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        all_embeddings = None
        gc.collect()
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        logger.debug(f"Created embeddings of shape {self.embeddings.shape}")
        logger.debug(f"FAISS index contains {self.index.ntotal} vectors")
    
    def retrieve_similiar_faqs(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant FAQs for a given query."""
        if self.index is None or self.faqs is None or self.embeddings is None:
            raise ValueError("Embeddings not created yet. Call create_embeddings first.")
        
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        relevant_faqs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.faqs):
                faq = self.faqs[idx].copy()
                similarity = float(1.0 / (1.0 + distances[0][i]))
                faq['similarity'] = similarity
                relevant_faqs.append(faq)
        
        return relevant_faqs

# Data Processing
def load_who_faq_data() -> List[Dict[str, Any]]:
    """Load mock WHO,CDC & PUBMed FAQ data."""
    logger.debug("Loading WHO,CDC & PUBMed medical FAQ data...")
    sample_faqs = [{"question":"What are the warning signs of a stroke?","answer":"Warning signs include sudden numbness or weakness, confusion, trouble speaking, vision problems, loss of balance, and severe headache.","category":"Noncommunicable Diseases","source":"CDC"},
                   {"question":"How is measles transmitted?","answer":"Measles spreads through respiratory droplets from coughing and sneezing and can remain infectious in the air for hours.","category":"Infectious Diseases","source":"WHO"},
                   {"question":"What are the recommended steps to prevent obesity?","answer":"Healthy eating habits, regular physical activity, behavior changes, and supportive environments help prevent obesity.","category":"Noncommunicable Diseases","source":"WHO"},{"question":"What is the treatment for depression?","answer":"Treatment often includes psychotherapy, medication like antidepressants, lifestyle changes, and social support.","category":"Mental Health","source":"WHO"},{"question":"How does HPV cause cancer?","answer":"High-risk HPV infections can cause changes in cells that may lead to cervical, anal, or throat cancers.","category":"Infectious Diseases","source":"CDC"},{"question":"What are the symptoms of asthma?","answer":"Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing, especially at night or early morning.","category":"Respiratory Diseases","source":"CDC"},{"question":"How can antibiotic resistance be prevented?","answer":"Appropriate use of antibiotics, infection prevention, vaccination, and good hygiene help prevent antibiotic resistance.","category":"Public Health","source":"WHO"},{"question":"What are the major causes of liver cirrhosis?","answer":"Chronic alcohol consumption, viral hepatitis B and C infections, and nonalcoholic fatty liver disease are major causes.","category":"Noncommunicable Diseases","source":"PubMed"},{"question":"What is the role of vaccines in preventing cervical cancer?","answer":"Vaccines against HPV prevent the majority of cervical cancers by targeting the most common cancer-causing virus strains.","category":"Vaccination and Immunization","source":"WHO"},{"question":"How is cholera transmitted?","answer":"Cholera is transmitted through contaminated water or food, often in areas with inadequate water treatment and sanitation.","category":"Infectious Diseases","source":"WHO"},{"question":"What is the best way to prevent foodborne illnesses?","answer":"Safe food handling, thorough cooking, proper storage, and good hygiene practices help prevent foodborne illnesses.","category":"Public Health","source":"CDC"},{"question":"How much physical activity is recommended for adults?","answer":"Adults should aim for at least 150 minutes of moderate-intensity or 75 minutes of vigorous-intensity physical activity per week.","category":"Healthy Lifestyle","source":"WHO"},{"question":"What are the long-term effects of COVID-19?","answer":"Long COVID symptoms may include fatigue, shortness of breath, cognitive dysfunction, and other complications lasting weeks or months.","category":"Infectious Diseases","source":"PubMed"},{"question":"How can you reduce the risk of developing Alzheimer's disease?","answer":"Managing cardiovascular risk factors, staying mentally active, regular exercise, and a healthy diet may lower the risk.","category":"Neurological Disorders","source":"CDC"},{"question":"What is antimicrobial stewardship?","answer":"Antimicrobial stewardship promotes appropriate use of antimicrobials to improve patient outcomes and reduce resistance development.","category":"Public Health","source":"WHO"},{"question":"How can heatwaves affect human health?","answer":"Heatwaves can cause heat exhaustion, heatstroke, dehydration, cardiovascular stress, and worsen pre-existing health conditions.","category":"Environmental Health","source":"WHO"},{"question":"What are effective ways to manage hypertension?","answer":"Lifestyle changes like reduced salt intake, regular exercise, healthy diet, weight loss, and medications are effective.","category":"Noncommunicable Diseases","source":"CDC"},{"question":"What is herd immunity?","answer":"Herd immunity occurs when a large portion of a population becomes immune to an infectious disease, indirectly protecting others.","category":"Vaccination and Immunization","source":"WHO"},{"question":"How do vaccines work in the body?","answer":"Vaccines stimulate the immune system to recognize and fight specific pathogens without causing disease themselves.","category":"Vaccination and Immunization","source":"WHO"},{"question":"What is the importance of prenatal care?","answer":"Prenatal care improves maternal health, monitors fetal development, and prevents pregnancy-related complications through early detection and intervention.","category":"Maternal Health","source":"WHO"},{"question":"What are the symptoms of COVID-19?","answer":"Common symptoms include fever, cough, fatigue, breathing difficulties, and loss of taste or smell.","category":"Infectious Diseases","source":"WHO"},{"question":"How can malaria be prevented?","answer":"Use insecticide-treated bed nets, indoor spraying with insecticides, and take preventive antimalarial medicines when traveling to endemic areas.","category":"Tropical Diseases","source":"WHO"},{"question":"What are the early signs of diabetes?","answer":"Early symptoms include frequent urination, excessive thirst, unexplained weight loss, fatigue, and blurred vision.","category":"Noncommunicable Diseases","source":"CDC"},{"question":"How is tuberculosis (TB) transmitted?","answer":"TB is transmitted from person to person through the air, typically when someone with active TB coughs or sneezes.","category":"Infectious Diseases","source":"WHO"},{"question":"What are effective strategies to prevent heart disease?","answer":"Maintain a healthy diet, exercise regularly, avoid tobacco use, and manage conditions like hypertension and diabetes.","category":"Noncommunicable Diseases","source":"CDC"},{"question":"What is the purpose of vaccination?","answer":"Vaccination protects individuals against infectious diseases by stimulating the immune system to recognize and fight pathogens.","category":"Vaccination and Immunization","source":"WHO"},{"question":"What are the main symptoms of measles?","answer":"Symptoms include high fever, cough, runny nose, inflamed eyes, and a characteristic red rash.","category":"Infectious Diseases","source":"CDC"},{"question":"How does antimicrobial resistance develop?","answer":"It develops when microorganisms evolve mechanisms to survive exposure to antimicrobial drugs, making treatments less effective.","category":"Public Health","source":"WHO"},{"question":"What are common symptoms of asthma?","answer":"Symptoms include wheezing, shortness of breath, chest tightness, and coughing, especially at night or early morning.","category":"Respiratory Diseases","source":"CDC"},{"question":"How can obesity be prevented?","answer":"Prevention includes maintaining a healthy diet, engaging in regular physical activity, and monitoring body weight regularly.","category":"Noncommunicable Diseases","source":"WHO"},{"question":"What steps help prevent the spread of influenza?","answer":"Get vaccinated annually, wash hands regularly, cover coughs and sneezes, and stay home when sick.","category":"Infectious Diseases","source":"CDC"},{"question":"What are risk factors for developing cancer?","answer":"Tobacco use, infections, radiation, poor diet, physical inactivity, and environmental pollutants are major risk factors.","category":"Noncommunicable Diseases","source":"WHO"},{"question":"How does hypertension affect health?","answer":"High blood pressure can damage arteries and organs over time, leading to heart disease, stroke, and kidney problems.","category":"Noncommunicable Diseases","source":"CDC"},{"question":"What is dengue fever and how is it transmitted?","answer":"Dengue fever is a mosquito-borne viral infection causing high fever, rash, and muscle pain, transmitted by Aedes mosquitoes.","category":"Tropical Diseases","source":"WHO"},{"question":"How can depression be managed?","answer":"Management involves psychotherapy, medications like antidepressants, and lifestyle changes such as exercise and social support.","category":"Mental Health","source":"CDC"},{"question":"What are the common signs of stroke?","answer":"Signs include sudden numbness, confusion, trouble speaking, vision problems, dizziness, and severe headache.","category":"Neurological Diseases","source":"CDC"},{"question":"What preventive measures reduce the risk of skin cancer?","answer":"Use sunscreen, wear protective clothing, seek shade, and avoid indoor tanning.","category":"Noncommunicable Diseases","source":"CDC"},{"question":"What treatments are available for Alzheimer's disease?","answer":"Current treatments help manage symptoms, including memory support, cognitive therapy, and medications to slow disease progression.","category":"Neurological Diseases","source":"PubMed"},{"question":"What is the significance of breastfeeding for newborns?","answer":"Breastfeeding provides essential nutrients and antibodies that protect against infections and promote healthy development.","category":"Maternal and Child Health","source":"WHO"},{"question":"How does physical activity benefit mental health?","answer":"Regular exercise reduces symptoms of depression and anxiety and improves mood and overall mental well-being.","category":"Mental Health","source":"PubMed"},{"question":"What causes antibiotic resistance in hospitals?","answer":"Overuse and misuse of antibiotics in healthcare settings lead to the development and spread of resistant bacteria.","category":"Public Health","source":"CDC"},{"question":"What are common symptoms of COVID-19 variants?","answer":"Symptoms of COVID-19 variants are similar to the original strain, including cough, fever, fatigue, but may vary slightly in severity.","category":"Infectious Diseases","source":"WHO"}]


    local_path = "data/medical_faqs.json"
    os.makedirs("data", exist_ok=True)
    with open(local_path, 'w') as f:
        json.dump(sample_faqs, f)
    logger.debug(f"Saved mock dataset to {local_path}, loaded {len(sample_faqs)} FAQs")
    return sample_faqs

def preprocess_faq(faqs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess FAQ data: clean text and filter invalid entries."""
    processed_faqs = []
    for faq in faqs:
        question = str(faq.get('question', '')).strip()
        answer = str(faq.get('answer', '')).strip()
        faq['question'] = question
        faq['answer'] = answer
        if question and answer:
            processed_faqs.append(faq)
        else:
            logger.warning(f"Skipping invalid FAQ: question='{question}', answer='{answer}'")
    logger.debug(f"After preprocessing: {len(processed_faqs)} valid FAQ entries")
    return processed_faqs

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, gc, psutil, logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "distilgpt2": "distilbert/distilgpt2",
    "gpt2": "openai-community/gpt2",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi3": "microsoft/Phi-3-mini-4k-instruct"
}

# Response Generation
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, gc, psutil, logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "distilgpt2": "distilbert/distilgpt2",
    "gpt2": "openai-community/gpt2",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "phi3": "microsoft/Phi-3-mini-4k-instruct"
}

class MedicalResponseGenerator:
    def __init__(self, model_name: str = "distilgpt2"):
        check_memory()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {self.device}")
        
        model_id = MODEL_REGISTRY.get(model_name.lower(), model_name)
        logger.info(f"Loading model: {model_name} → {model_id}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map={"": "cpu"},
                    torch_dtype=torch.float32
                )

            # Handle tokenizer padding
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

            logger.debug("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise RuntimeError(f"Failed to load {model_name} model.")

    def create_prompt(self, query: str, faqs: List[Dict[str, str]]) -> str:
        examples = "\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs])
        return f"{examples}\nQ: {query}\nAnswer:"

    def generate_response(self, query: str, relevant_faqs: List[Dict[str, Any]]) -> str:
        prompt = self.create_prompt(query, relevant_faqs)
        logger.debug(f"Prompt:\n{prompt}")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.5,
                top_p=0.95,
                do_sample=True,
                num_beams=3,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.pad_token_id
            )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        response = output.split("Answer:", 1)[-1].strip()
        if len(response) < 10:
            response = "The requested medical information is unavailable or insufficient. Please consult a professional."
        
        return response

# class MedicalResponseGenerator:
#     def __init__(self, model_name: str = "distilbert/distilgpt2"):
#         """Initialize the response generator with DistilGPT2."""
#         check_memory()
#         logger.debug(f"Loading LLM: {model_name}")
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.debug(f"Using device: {self.device}")
        
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             gc.collect()
#             if self.device == "cuda":
#                 torch.cuda.empty_cache()
            
#             if self.device == "cuda":
#                 quantization_config = BitsAndBytesConfig(
#                     load_in_4bit=True,
#                     bnb_4bit_compute_dtype=torch.float16,
#                     bnb_4bit_use_double_quant=True,
#                     bnb_4bit_quant_type="nf4"
#                 )
#                 available_memory = psutil.virtual_memory().total / (1024 ** 3)
#                 gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
#                 max_memory = {0: f"{min(gpu_memory, 15)}GiB", "cpu": f"{min(available_memory, 30)}GiB"}
#                 logger.debug(f"Setting max_memory: {max_memory}")
                
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_name,
#                     quantization_config=quantization_config,
#                     device_map="auto",
#                     torch_dtype=torch.float16,
#                     #max_memory=max_memory
#                 )
#             else:
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_name,
#                     device_map={"": "cpu"},
#                     torch_dtype=torch.float32
#                 )
#             logger.debug("LLM loaded successfully")
#         except Exception as e:
#             logger.error(f"Model loading error for {model_name}: {str(e)}")
#             logger.info("Falling back to gpt2...")
#             try:
#                 model_name = "openai-community/gpt2"
#                 self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     model_name,
#                     device_map={"": self.device},
#                     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
#                 )
#                 logger.debug("Fallback LLM loaded successfully")
#             except Exception as fallback_e:
#                 logger.error(f"Fallback model loading error: {str(fallback_e)}")
#                 raise RuntimeError("Failed to load both primary and fallback LLMs")

#     def generate_response(self, query: str, relevant_faqs: List[Dict[str, Any]]) -> str:
        
#         prompt = self.create_prompt(query, relevant_faqs)
#         logger.debug(f"Prompt sent to model:\n{prompt}")
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
#         # Ensure pad_token_id is set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=150,
#                 temperature=0.5,
#                 top_p=0.95,
#                 do_sample=True,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 num_beams=3,
#                 no_repeat_ngram_size=2
#             )
        
#         raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logger.debug(f"Raw model output:\n{raw_response}")

#         marker ="Answer:"
#         if marker in raw_response:
#             response=raw_response.split(marker,1)[1].strip()
#         else:
#             logger.warning("Answer: marker not found in raw response")
#             response = raw_response[len(prompt):].strip()

#         # Remove FAQ markers, artifacts, and repetitive patterns
#         response = response.replace("Q:", "").replace("A:", "").replace("", "")
#         # Remove any lines starting with FAQ-like patterns
#         response_lines = [line for line in response.split("\n") if not line.strip().startswith(("FAQ:", "Q:", "A:"))]
#         response = " ".join(response_lines).strip()
        
#         # Fallback if response is empty or too short
#         if not response or len(response) < 10:
#             logger.warning("Generated response is empty or too short, using fallback")
#             response = "The warning signs of a stroke include sudden numbness or weakness, confusion, trouble speaking, vision problems, loss of balance, and severe headache. Please consult a doctor for medical advice."
        
#         logger.debug(f"Processed response:\n{response}")    
#         ############    
        
#         if self.device == "cuda":
#             torch.cuda.empty_cache()
        
#         return response    

    ####ORG##
    # def generate_response(self, query: str, relevant_faqs: List[Dict[str, Any]]) -> str:
        
    #     prompt = self.create_prompt(query, relevant_faqs)
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
    #     with torch.no_grad():
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_new_tokens=150,
    #             temperature=0.7,
    #             top_p=0.9,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
        
    #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     response = response[len(prompt):].strip()
        
    #     if self.device == "cuda":
    #         torch.cuda.empty_cache()
        
    #     return response
    
    def create_prompt(self, query: str, relevant_faqs: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM with retrieved FAQs as context."""
        faq_context = "\n\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in relevant_faqs])
        logger.debug(f"FAQ context for query '{query}':\n{faq_context}")
        prompt = f"""
You are a medical assistant powered by WHO data. Below are relevant WHO medical FAQ entries in English:

{faq_context}

Based on the information above, provide a helpful, accurate, and concise response to the following medical query.Respond only in English. If the query is not covered by the FAQs, provide a general response based on WHO guidelines, but note that you are not a doctor and recommend consulting one for specific medical advice.

Medical Query: {query}

Response: """
        return prompt

# Main Chatbot Class
class WHOMedicalRAGChatbot:
    def __init__(self):
        """Initialize the WHO Medical RAG Chatbot."""
        self.faqs = preprocess_faq(load_who_faq_data())
        self.embedder = MedicalFAQEmbedder()
        self.embedder.create_embeddings(self.faqs)
        self.response_generator = MedicalResponseGenerator()
    
    @time_function
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer a medical query using RAG."""
        relevant_faqs = self.embedder.retrieve_similiar_faqs(query, k=3)
        response = self.response_generator.generate_response(query, relevant_faqs)
        evaluation = evaluate_response(response)
        memory_stats = memory_statics()
        gen = MedicalResponseGenerator(model_name="llama3")  # or "mistral", "phi3", etc.
        response = gen.generate_response("What are the signs of a stroke?", relevant_faqs)
        print(response)

        
        return {
            "query": query,
            "response": response,
            "relevant_faqs": relevant_faqs,
            "evaluation": evaluation,
            "memory_stats": memory_stats
        }

# Flask Routes
@app.route('/')
def index():
    """Render the main chatbot page."""
    template_path = os.path.join('templates', 'index.html')
    if not os.path.exists(template_path):
        logger.error(f"Template not found at {template_path}")
        return "Error: Template not found", 500
    logger.debug("Serving index.html")
    return render_template('index.html')



@app.route('/query', methods=['POST'])
def handle_query():
    """Handle query from the web interface."""
    try:
        # Get the query and model from the request body
        data = request.get_json()
        query = data.get('query', '').strip()
        model_name = data.get('model', 'distilbert/distilgpt2')  # Default to DistilGPT-2
        
        if not query:
            logger.warning("Empty query received")
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Initialize the response generator with the selected model
        gen = MedicalResponseGenerator(model_name=model_name)
        
        # Retrieve relevant FAQs using the embedder
        relevant_faqs = chatbot.embedder.retrieve_similiar_faqs(query, k=3)  # Assuming this method exists
        
        response = gen.generate_response(query, relevant_faqs)
        
        # Prepare the result
        result = {
            "query": query,
            "response": response,
            "relevant_faqs": relevant_faqs,
            "evaluation": evaluate_response(response),  # Assuming you have an evaluation function
            "memory_stats": memory_statics()  # Assuming you have a memory stat function
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

# HTML Template
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WHO Medical RAG Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }
        h1 { color: #333; text-align: center; }
        .chat-container { max-width: 800px; margin: auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .input-box { width: 100%; padding: 10px; margin-bottom: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 4px; }
        .submit-btn { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .submit-btn:hover { background: #0056b3; }
        .response-area { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 4px; }
        .faq-item { margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>Healthcare & Medical Knowledge RAG Chatbot</h1>
    <div class="chat-container">
        <!-- Model Selection Dropdown -->
        <label for="modelSelect">Select Model:</label>
        <select id="modelSelect" class="input-box">
            <option value="distilbert/distilgpt2">DistilGPT-2</option>
            <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral 7B</option>
            <option value="meta-llama/Meta-Llama-3-8B-Instruct">LLaMA 3</option>
            <option value="microsoft/phi-2">Phi-2</option>
        </select>

        <!-- Query Input -->
        <input type="text" id="queryInput" class="input-box" placeholder="Enter your medical query">
        
        <!-- Submit Button -->
        <button class="submit-btn" onclick="submitQuery()">Submit</button>

        <!-- Response Area -->
        <div id="responseArea" class="response-area"></div>
    </div>

    <script>
        async function submitQuery() {
            const queryInput = document.getElementById('queryInput');
            const modelSelect = document.getElementById('modelSelect');
            const responseArea = document.getElementById('responseArea');
            
            const query = queryInput.value.trim();
            const modelName = modelSelect.value;  // Get the selected model

            if (!query) {
                responseArea.innerHTML = `<p class="error">Please enter a query.</p>`;
                return;
            }

            responseArea.innerHTML = `<p>Loading...</p>`;

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, model: modelName })
                });
                const data = await response.json();

                if (data.error) {
                    responseArea.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                    return;
                }

                let html = `<h3>Query: ${data.query}</h3>`;
                html += `<p><strong>Response:</strong> ${data.response}</p>`;
                html += `<h4>Relevant FAQs:</h4>`;
                data.relevant_faqs.forEach(faq => {
                    html += `<div class="faq-item">
                        <p><strong>Q:</strong> ${faq.question}</p>
                        <p><strong>A:</strong> ${faq.answer}</p>
                        <p><strong>Similarity:</strong> ${(faq.similarity * 100).toFixed(2)}%</p>
                    </div>`;
                });
                html += `<h4>Evaluation:</h4>`;
                html += `<p>Length: ${data.evaluation.length} characters</p>`;
                html += `<p>Word Count: ${data.evaluation.word_count}</p>`;
                html += `<h4>Memory Stats:</h4>`;
                for (const [key, value] of Object.entries(data.memory_stats)) {
                    html += `<p>${key}: ${value}</p>`;
                }
                

                responseArea.innerHTML = html;
            } catch (error) {
                responseArea.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            }
        }

        // Allow Enter key to submit query
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                submitQuery();
            }
        });
    </script>
</body>
</html>
"""

# Save HTML Template
template_dir = "templates"
data_dir = "data"
logger.debug(f"Current working directory: {os.getcwd()}")
os.makedirs(template_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
template_path = os.path.join(template_dir, "index.html")
with open(template_path, "w") as f:
    f.write(index_html)
logger.debug(f"Saved index.html to {template_path}")

# Initialize Chatbot
try:
    logger.debug("Initializing WHOMedicalRAGChatbot...")
    chatbot = WHOMedicalRAGChatbot()
    logger.debug("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    raise

# Run Flask App
if __name__ == "__main__":
    logger.debug("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)