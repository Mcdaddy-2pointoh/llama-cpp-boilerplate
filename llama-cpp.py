# Imports
from llama_cpp import Llama

# Instanciate the model
my_aweseome_llama_model = Llama(model_path="./models/Hermes-3-Llama-3.1-8B/Hermes-3-Llama-3.1-8B.Q4_K_M.gguf")


prompt = "This is a prompt"
max_tokens = 100
temperature = 0.3
top_p = 0.1
echo = True
stop = ["Q", "\n"]

# Define the parameters
model_output = my_aweseome_llama_model(
       prompt,
       max_tokens=max_tokens,
       temperature=temperature,
       top_p=top_p,
       echo=echo,
       stop=stop,
   )
final_result = model_output["choices"][0]["text"].strip()

print(final_result)