# pjjob

 https://docs.conda.io/en/latest/miniconda.html
 
 conda create -n basic_model python=3.6.8 fastapi uvicorn python-dotenv pydantic locust plotly scikit-learn seaborn jupyter -c conda-forge

first
conda activate basic_model

second
uvicorn api:app --host 0.0.0.0 --port 80 --reload
