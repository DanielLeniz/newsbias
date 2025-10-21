from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BIAS_MODEL_NAME = os.getenv("BIAS_MODEL_NAME", "")

DATA_DIR = os.getenv("DATA_DIR", "data")
ALLSIDES_PRIORS_PATH = os.getenv("ALLSIDES_PRIORS_PATH", os.path.join(DATA_DIR, "allsides_priors.csv"))

QBIAS_CSV = os.path.join(DATA_DIR, "./data/qbias_articles.csv")

SUMMARY_ENABLED = os.getenv("SUMMARY_ENABLED", "1") == "1"
HF_ENABLED = os.getenv("HF_ENABLED", "0") == "1"
