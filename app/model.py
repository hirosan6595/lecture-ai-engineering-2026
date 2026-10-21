uv run python -c "
from google import genai
from app.config import get_api_key

client = genai.Client(api_key=get_api_key())

for model in client.models.list():
    print(model.name)
"