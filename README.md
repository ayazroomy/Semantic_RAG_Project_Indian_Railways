### Semantic_RAG_Project_Indian_Railways

#### Sources:

- data/Indian_Railways_Annual_Report _23_24.pdf
- data_set/train_info.csv

#### Description:
In this Project we are taking two sources one pdf file and one csv file to build a RAG based Agents to run queries and gives response.This is build entirely using llamaindex and openai and chromadb to store the data and embeddings & retrivel.

We have also Utilize two QueryEngines JSONQueryEngine and llamad Index Query Engine to perform Route based Quering depend upon the 
nature of Query.

To run this Project Locally as an API:
---------
1. Add your open api key in .env file
2. Install uvicorn globally in the system.
3. Setup the Virtual Environment
4. run ==> pip install -r requirements.txt
5. run ==> uvicorn main:app to launch the fast api server
6. visit localhost:8000/docs to run the swagger page to test api


Live Demo : of this RAG Project (Railways AnnualReport 23-34 + Train Details)
-------------------------
DEmo Link:  https://ayazroomy-my-rag-space-railway.hf.space/webapp/index.html