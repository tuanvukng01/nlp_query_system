
# NLP-Driven Query Categorization and Response System with GPU Optimization 🚀

![Project Banner](https://img.shields.io/badge/NLP-GPU%20Optimized-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## **Introduction**

This project focuses on creating an **NLP-based Query Categorization and Response System** optimized for GPUs. The system leverages **state-of-the-art transformer models** and integrates multiple features such as:

- Query categorization into predefined groups (e.g., Billing, Technical Support, General Inquiry).
- Sentiment analysis (Positive, Neutral, or Negative sentiment).
- Automatic response generation using templates linked to query categories and sentiments.
- GPU acceleration for faster training and inference.
- A scalable RESTful API interface for real-time query processing.

---

## **Features**
- **Query Categorization**: Fine-tune transformer-based models (e.g., BERT) for multi-class categorization.
- **Sentiment Analysis**: Predict sentiment labels with high accuracy.
- **Response Generation**: Generate contextual responses using predefined templates.
- **Synthetic Data Generation**: Mock data for testing with randomized variability.
- **GPU Optimization**: Mixed-precision training using PyTorch and CUDA.
- **Scalable Deployment**: Dockerized for local testing; deployable via Kubernetes.
- **RESTful API**: Expose the system via endpoints for real-time processing.

---

## **Project Directory Structure**

```
nlp_query_system/
├─ data/
│  ├─ mock_data_generation.py           
│  ├─ synthetic_data.json               
│  ├─ synthetic_data.csv               
├─ models/
│  ├─ categorization_model.py         
│  ├─ sentiment_model.py               
│  ├─ transformer_finetune.py          
│  ├─ utils.py                          
├─ train/
│  ├─ train_category.py             
│  ├─ train_sentiment.py               
│  ├─ train_config.yaml                 
├─ inference/
│  ├─ inference_pipeline.py             
│  ├─ response_templates.py             
├─ api/
│  ├─ app.py                            
│  ├─ Dockerfile                      
│  ├─ requirements.txt                 
│  ├─ gunicorn_config.py               
│  ├─ swagger.yaml                  
│  ├─ k8s_deployment.yaml              
├─ scripts/
│  ├─ run_docker.sh                   
│  ├─ deploy_k8s.sh                    
│  ├─ download_models.sh              
│  ├─ generate_data.sh              
└─ README.md                           
```

---

## **Technologies Used**

- **Programming Languages**: Python 3.10+
- **NLP Libraries**: Hugging Face Transformers, NLTK, Spacy
- **Deep Learning**: PyTorch (with CUDA)
- **Data Processing**: Pandas, NumPy
- **API Development**: Flask, Swagger
- **Containerization**: Docker
- **Orchestration**: Kubernetes

---

## **Setup Instructions**

### **Pre-requisites**
1. Python 3.10+ installed.
2. GPU-enabled machine with CUDA installed.
3. Docker and Kubernetes installed.

### **Installation Steps**

1. Clone the repository:
   ```bash
   git clone https://github.com/tuanvukng01/nlp_query_system.git
   cd nlp_query_system
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r api/requirements.txt
   ```

3. Generate mock data:
   ```bash
   python data/mock_data_generation.py
   ```

4. Train the models:
   ```bash
   python train/train_category.py
   python train/train_sentiment.py
   ```

5. Run the API locally:
   ```bash
   python api/app.py
   ```

6. Build and run Docker container:
   ```bash
   bash scripts/run_docker.sh
   ```

7. Deploy with Kubernetes:
   ```bash
   bash scripts/deploy_k8s.sh
   ```

---

## **Endpoints**
- **`POST /categorize`**: Categorize a query.
- **`POST /sentiment`**: Analyze sentiment.
- **`POST /response`**: Generate a category- and sentiment-specific response.

Refer to the Swagger documentation (`api/swagger.yaml`) for detailed usage.

---

[//]: # (## **Screenshots**)

[//]: # ()
[//]: # (![Mock Data Generation]&#40;https://img.icons8.com/color/48/000000/code.png&#41;)

[//]: # (_Sample mock data generation script in action._)

[//]: # ()
[//]: # (---)

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request for review.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

[//]: # (---)

[//]: # ()
[//]: # (## **Contact**)

[//]: # (- Author: [Your Name])

[//]: # (- Email: your.email@example.com)

[//]: # (- LinkedIn: [Your LinkedIn Profile])
