### Semantic_RAG_Project_Indian_Railways

#### Sources:

- data/Indian_Railways_Annual_Report _23_24.pdf
- data_set/train_info.csv

#### Description:
In this Project we are taking two sources one pdf file and one csv file to build a RAG based Agents to run queries and gives response.This is build entirely using llamaindex and openai and chromadb to store the data and embeddings.

To run this Project Locally as an API:
---------
1. Add your open api key in .env file
2. Install uvicorn globally in the system.
3. run ==> pip install -r requirements.txt
4. run ==> uvicorn main:app to launch the fast api server
5. visit localhost:8000/docs to run the swagger page to test api