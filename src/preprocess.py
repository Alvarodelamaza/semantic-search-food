import json
import openai
from dotenv import load_dotenv
import os 

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)



def generate_test(texts, model="gpt-4.1-mini", batch_size=10):
    test_set = {}
    responses = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Here are some examples of search queries in Portuguese: Prato de proteína mista, Tigela de sopa quentinha para o inverno, Arroz temperado do norte da África, Prato para almoço de escritório, Caixa de proteína rápida e saudável, Carne assada caseira com acompanhamentos, Lanche reconfortante para a madrugada"
                },
                {"role": "user", "content": f"Generate one hypothetical, realistic, not too detailed query, with out name brands or quantity, for each of the items in the list, between 4 and 6 words more or less. LIST OF PRODUCTS: {batch}"}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "generate_queries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": ["queries"]
                    }
                }
            }],
            tool_choice={"type": "function", "function": {"name": "generate_queries"}}
        )
        
        #Extract data
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        responses.extend(result["queries"])
    
    return responses


def preprocess_metadata(item_metadata, item_profile,type):

    metadata = json.loads(item_metadata)
    profile = json.loads(item_profile)
    
    # Combine text
    if type== "Basic":
        text = f"{metadata['name']} {metadata['description']} {metadata['category_name']} {metadata['taxonomy']['l0']} {metadata['taxonomy']['l1']} {metadata['taxonomy']['l2']}"

    # Combine Natural text
    if type == "Natural":
        text = f"{metadata['name']} pertencente à categoria {metadata['category_name']} em format {metadata['description']} classificado como {metadata['taxonomy']['l0']}, {metadata['taxonomy']['l1']} e {metadata['taxonomy']['l2']}"
    
    
    # Add co-purchase item names
    #co_purchase_items = profile.get('metrics', {}).get('coPurchaseItems', [])
    ##text += f" {co_purchase_names}"
    
    return text


