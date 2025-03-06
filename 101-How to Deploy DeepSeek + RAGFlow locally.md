How to Deploy DeepSeek + RAGFlow locally

**Reason**

- **personal knowledge storage**
- **the limit of files uploaded**
- **the connection of the context**

**Practical Example**

 A man who read a lot of books on American literature, and when I asked he the question "How do u know about 史蒂夫金"， he would give a answer based on what he read. And if I want to learn some knowledge about King, he would say some nonsense.

**Framework**

- DeepSeek Deployed locally: quiz and chat
- RAG: construct personal knowledge storage
- Embedding: deal with the input texts

**Keypoinqs**

- **RAG** (Retrieval-Augmented Generation) 
   
   - **Retrieval: search for the relevant content with the question**
   - **Augmentation: combine the input with the infomation retrieved**
   - **Generation: based on the input, the model capitapility and the retrieval external information**

- **Embedding Model**

  ![CleanShot 2025-02-24 at 16.14.15@2x](https://cdn.statically.io/gh/stoneBuild29/MyPictures@main/upload/CleanShot%202025-02-24%20at%2016.14.15%402x.png)

  - Analyze the natural language of the text
  - Map it to high-dimensional vectors
  - Capture the similarity between different texts

- **Ollama**
  - a platform to run and manage LLM 
  - env config
    - OLLAMA HOST - 0.0.0.0:11434
    - OLLAMA MODELS- defined positon

- **Docker**
  - set the running dependency and libraries

**Process**

> 把冰箱门打开
>
> 把大象装进去
>
> 把冰箱门关上

- Download Ollama and DeepSeek locally (32b/1.5b)
- Download RAGflow and Docker
  - Download RAGflow source code
  - Download Docker
- Build a personal storage within the RAGflow

**Web**

1. https://deepseek.csdn.net/67c3a00d3b685529b702ff03.html
2. 