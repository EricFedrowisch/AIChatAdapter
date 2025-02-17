# AIChatAdapter

AIChatAdapter is a powerful tool that simplifies integrating and switching  
between multiple Large Language Model (LLM) API providers, such as OpenAI,  
Anthropic, Ollama, and LM Studio, with minimal code changes. This adapter  
provides a single, consistent interface, allowing you to focus on building  
your application without getting bogged down in the specifics of each LLM service.  

## Use Cases

-Simplify your code by decoupling the LLM backend from the rest of your code  
-Easily test various LLMs against one another for quality, speed, etc  
-Have less downtime for critical LLM services via robust failover
systems that swap LLM backends if there is an outtage of one provider  

## Installation
-Clone this repository  
-Install the necessary python packages  
```
# Install these packages
python3 -m pip install openai dotenv

# Optionally install these packages if you plan on using the LLM APIs they provide
python3 -m pip install anthropic
python3 -m pip install ollama
```
-Create a new folder for your code and copy the .env config file there.  
-Put your api keys in the .env file.  
-Now you are ready to go.  

## Usage

### Example using OpenAI, the default backend
Here's an example of how to get a chat completion response from OpenAI.  
```python
from AIChatAdapter import AIChatAdapter  

# Initialize the Chat Adapter with a system prompt (OpenAI is default backend)  
client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="openai")  
# Get a response. All calls are asynchronous  
response = client.get_response("What is the capital of France?")  
# Print the chat messages.
for msg in client.chat.messages:  
    print(msg)  
```
This should print out something like this:  
```bash
{'role': 'system', 'content': 'Answer as best as you can.'}
{'role': 'user', 'content': 'What is the capital of France?'}
{'role': 'assistant', 'content': 'The capital of France is Paris.'}
```

### Example using the Anthropic backend.
Nearly identical to default example, except using Anthropic as the backend.  
```python
from AIChatAdapter import AIChatAdapter

# Initialize the Chat Adapter with Anthropic as backend
client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="anthropic")
response = client.get_response("What is the capital of France?")
for msg in client.chat.messages:
    print(msg)
```
This will print something like:  
```bash
{'role': 'user', 'content': 'Answer as best as you can.'}
{'role': 'user', 'content': 'What is the capital of France?'}
{'role': 'assistant', 'content': 'The capital of France is Paris.\n\nParis has been the capital city of France since 987 CE, when Hugh Capet made it the capital of his kingdom. It is located in the north-central part of the country on the Seine River.\n\nParis is not only the political capital but also the cultural, economic, and educational center of France. It is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Arc de Triomphe. The city is also famous for its cuisine, fashion, art, and overall cultural significance.\n\nWith a population of over 2 million in the city proper and over 12 million in the metropolitan area, Paris is the largest city in France and one of the most populous urban areas in Europe.'}
```
A little bit more verbose than OpenAI.  

### Example using Ollama on local machine
First, make sure you are running the [Ollama app](https://ollama.com/download).  
You will also need a model. The default model is Llama3. This can be changed in the configs in your .env file.  
You can check that you have the model downloaded for ollama with the terminal command below:  
```bash
# Check that you have Llama3 model
ollama pull llama3
```
Once you have Ollama running and the model downloaded, you can get a response with this python code.  
```python
from AIChatAdapter import AIChatAdapter

client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="ollama")
response = client.get_response("What is the capital of France?")
for msg in client.chat.messages:
    print(msg)
```
From this I got the response:  
```bash
{'role': 'system', 'content': 'Answer as best as you can.'}
{'role': 'user', 'content': 'What is the capital of France?'}
{'role': 'assistant', 'content': 'The capital of France is Paris.'}
```

### Example with LM Studio as local server
-Download and install [LM Studio](https://lmstudio.ai/).  
-Run LM Studio.  
-Click "Search" on the side bar. This will let you search for models to download.  
-To use the default model choice, copy and paste into the search bar:  
```bash
Meta-Llama-3-8B-Instruct-GGUF
```
-Hit the "Go" button and download the model.  
-Wait until the model is downloaded.  
-Click "Local Server" in the side bar.  
-At the top of that view, click "Select Model to Load" and load your model of choice.  
-Once the progress bar is complete, the local server should be up  
  
The code to get a response is basically identical except for the backend specified:  
```python
from AIChatAdapter import AIChatAdapter

model = "Meta-Llama-3-8B-Instruct-GGUF"
client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="local", model=model)
response = client.get_response("What is the capital of France?")
for msg in client.chat.messages:
    print(msg)
```
You can watch the response tokens being generated in the window at the bottom left of the "Local Server" view in LM Studio.  
Most of the local models I've tested with are a bit over verbose, so you may want to limit their token generation length for
faster replies.  