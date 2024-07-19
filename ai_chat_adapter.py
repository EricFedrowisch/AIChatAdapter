"""
Written by Eric Fedrowisch Apr. 2024. All rights reserved. License is CC BY-NC-ND 4.0 International.
"""

#Std Lib Imports
import asyncio
import copy
import logging
import os
from typing import Callable, Dict, List, Optional
# 3rd Party Imports
from dotenv import load_dotenv
from openai import AsyncOpenAI  # Needed for OpenAI as well as local LM Studio chats
# from anthropic import AsyncAnthropic  # Loaded if/when needed
# from ollama import AsyncClient  # Loaded if/when needed

# Environmental Inits
log = logging.getLogger() # Get local ref for logger
load_dotenv() # Load environment variables from .env file

# Default models are here for easy modification and updating.
openai_default_model = os.getenv("openai_default_model")
anthropic_default_model = os.getenv("anthropic_default_model")
local_default_model = os.getenv("local_default_model")
ollama_default_model = os.getenv("ollama_default_model")
# Local LMStudio Settings
lm_studio_base_url = os.getenv("lm_studio_base_url")


class AIChatAdapter:
    """Manages the AI client connection to get chat completion responses. Default AI backend is OpenAI."""
    def __init__(self, system_prompt:str = "", tools: List[Dict] = [], backend: str = "openai", model: str = "default"):
        try:
            self.system_prompt = system_prompt  # Store initial prompt
            self.chat = ChatBuffer(system_prompt=system_prompt)  # Init chat buffer object
            self.tools = tools  # Store tools schema ref
            # Create ai api client wrapper based on backend
            match backend.lower():
                case "openai":
                    self.client = OpenAIWrapper(model=model)
                case "anthropic":
                    self.client = AnthropicWrapper(model=model)
                    #Adjust tools schema for anthropic self.tools
                    if tools:
                        self.tools = self.client.adapt_tool_schema(tools)
                    # Adjust chat buffer for Anthropic
                    self.chat.messages[0]["role"] = "user"  # Anthropic errors if user is called "system"
                case "ollama":
                    self.client = OllamaWrapper(model=model)
                case "local":
                    self.client = LocalWrapper(model=model)
                case _:
                    raise Exception(f"Exception initializing AIChatAdapter: Backend Argument matches no available backend type.")
        except Exception as e:
            raise Exception(f"Exception initializing AIChatAdapter: {e}")

    async def get_response(self, message: str = "", role: str = "user") -> str:
        """Get a single response string from a client."""
        try:
            self.chat.update(content=message, role=role)
            response = await self.client.get_response(
                messages=self.chat.messages,
                tools=self.tools,
                )
            self.chat.update(**response)
            return response
        except Exception as e:
            raise Exception(f"Exception getting ai response: {e}")
        return None

    async def get_stream(self, message: str = "", role: str = "user", tools: List[Dict] = []):
        """Get a single response as a stream from a chat completion api call."""
        try:
            self.chat.update(content=message, role=role)
            stream = await self.client.get_stream(
                messages=self.chat.messages,
                tools=self.tools,
            )
            self.chat.update(**stream)
        except Exception as e:
            raise Exception(f"Exception getting ai stream: {e}")


class OpenAIWrapper:
    """AI API client wrapper for OpenAI chat completion requests."""
    def __init__(self, model:str = openai_default_model):
        if model == "default":  # Use default model from var "openai_default_model"
            self.model = openai_default_model
        else:
            self.model = model
        try:
            api_key = os.getenv("openai")  # Load API key
            if api_key is not None:
                self.client = AsyncOpenAI(api_key=api_key)  # Init AI Client
        except Exception as e:
            raise Exception(f"Exception creating OpenAIWrapper: {e}")

    async def get_response(self, messages: List[Dict], tools: List[Dict] = []) -> str | None:
        """Get a single response data object from a chat completion api call."""
        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                tools=tools if tools else None,  # OpenAI doesn't accept empty tools array here...
            )
            return {"role": response.choices[0].message.role, "content": response.choices[0].message.content}
        except Exception as e:
            raise Exception(f"Exception getting OpenAI response: {e}")
        return None

    # TODO: Need to make a robust StreamWrapper class that can actually yield chunks.
    async def get_stream(self, messages: List[Dict], tools: List[Dict] = []):
        """Get a single response as a stream from a chat completion api call."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
            msg = ""
            async for chunk in stream:
                msg = msg + (chunk.choices[0].delta.content or "")  # print(chunk.choices[0].delta.content or "", end="")
            return {"role": "assistant", "content": str(msg)}
        except Exception as e:
            raise Exception(f"Exception getting OpenAI stream: {e}")

class AnthropicWrapper:
    """AI API client wrapper for Anthropic message requests."""
    def __init__(self, model:str = anthropic_default_model):
        if model == "default":  # Use default model from var "anthropic_default_model"
            self.model = anthropic_default_model
        else:
            self.model = model
        try:
            from anthropic import AsyncAnthropic
            api_key = os.getenv("anthropic")  # Load API key
            if api_key is not None:
                self.client = AsyncAnthropic(api_key=api_key)  # Init AI Client
        except Exception as e:
            raise Exception(f"Exception creating AnthropicWrapper: {e}")

    def adapt_tool_schema(self, tools: List[Dict] = []):
        """Adjust a tools schema for Anthropic."""
        # Anthropic names the functions' "parameters" field "input_schema" instead...
        if tools:  # If tools is not empty...
            # Adapt "parameters" field to "input_schema"
            adapted = copy.deepcopy(tools)  # Deep copy to avoid modifying original. May be overkill.
            for schema in adapted:
                fx = schema["function"]  # Get function dict
                fx["input_schema"] = fx.pop("parameters")  # Reassign and rename popped key
            return adapted
        else:  # If tools IS empty...
            return tools  # Return empty. No adaptation needed.

    async def get_response(self, messages: List[Dict], tools: List[Dict] = []) -> str | None:
        """Get a single response data object from a messages api call."""
        response = None
        messages = self.merge_consecutive_messages(messages)
        try:
            kwargs = {  # More complicated approach needed here to handle empty tools list
                "messages":messages,
                "model":self.model,
                "max_tokens":1024,
            }
            if tools:
                kwargs["tools"]=tools
            message = await self.client.messages.create(**kwargs)
            response = {"role": message.role, "content": str(message.content[0].text)}
        except Exception as e:
            raise Exception(f"Exception getting Anthropic response: {e}")
        return response

    def merge_consecutive_messages(self, messages: List[Dict]) -> List[Dict]:
        """Create role alternating consecutive messages for Anthropic."""
        if not messages:
            return []
        merged_messages = []
        current_message = messages[0].copy()
        for i in range(1, len(messages)):
            if messages[i]['role'] == current_message['role']:
                current_message['content'] += ' ' + messages[i]['content']
            else:
                merged_messages.append(current_message)
                current_message = messages[i].copy()
        # Append the last message
        merged_messages.append(current_message)
        return merged_messages


class LocalWrapper:
    """AI API client wrapper for getting chat completions from a local LM Studio server."""
    def __init__(self, model:str = local_default_model):
        # Point to the local server
        if model == "default":
            self.model = local_default_model
        else:
            self.model = model
        try:
            self.client = AsyncOpenAI(base_url=lm_studio_base_url, api_key="lm-studio")
        except Exception as e:
            raise Exception(f"Exception creating LM Studio local LLM client: {e}")

    async def get_response(self, messages: List[Dict], tools: List[Dict] = []) -> str | None:
        """Get a single response data object from a local LLM server chat completion api call."""
        try:
            response = await self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                # tools=tools,  # TODO: Local tools support possible?
            )
            return {"role": response.choices[0].message.role, "content": response.choices[0].message.content}
        except Exception as e:
            raise Exception(f"Exception getting LM Studio response: {e}")
        return None


class OllamaWrapper:
    def __init__(self, model:str = ollama_default_model):
        if model == "default":
            self.model = ollama_default_model
        else:
            self.model = model
        try:
            from ollama import AsyncClient
            self.client = AsyncClient()
        except Exception as e:
            raise Exception(f"Exception creating OllamaWrapper: {e}")

    async def get_response(self, messages: List[Dict], tools: List[Dict] = []) -> str | None:
        response = await self.client.chat(model=self.model, messages=messages)
        message = response["message"]
        return message

class ChatBuffer:
    def __init__(self, system_prompt: str = '', messages: list[dict] = []):
        """Create a chat message buffer. Message bufers are lists of dicts containing user str keys and content str values."""
        # Create initial message buffer
        self.messages = [{"role": "system", "content": system_prompt}]  # Initial system prompt for high level context (ie 'You are an AI GM')
        self.messages.extend(messages)  # Extend buffer with additional optional messages to import them ie

    def update(self, content: str = "", role: str = "user"):
        """Update internal chat buffer state with AI chat response."""
        self.messages.append({"role": role, "content": content})

    def __repr__(self):
        repr = ""
        for msg in self.messages:
            repr = repr + f"{msg['role']}: {msg['content']}\n"
        return repr


if __name__ == "__main__":
    pass
    # Code Examples. Uncomment to run.
    # ChatBuffer Example.
    # chat = ChatBuffer(system_prompt="Foo and then bar, pls.")
    # chat.update(content="No, you foo.", role="user")
    # print(chat)

    # Local LM Studio Example
    # print(f"Local LLM Example Response (LM Studio running {local_default_model}):")
    # local_client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="local")
    # loop = asyncio.get_event_loop()
    # response = loop.run_until_complete(local_client.get_response("What is the capital of France?"))
    # for msg in local_client.chat.messages:
    #     print(msg)

    # OpenAI Example
    # print("\nOpenAI Example Response:")
    # oai_client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="openai")
    # loop = asyncio.get_event_loop()
    # response = loop.run_until_complete(oai_client.get_response("What is the capital of France?"))
    # for msg in oai_client.chat.messages:
    #     print(msg)

    # OpenAI Stream Example
    # print("\nOpenAI Example Stream:")
    # oai_client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="openai")
    # loop = asyncio.get_event_loop()
    # stream = loop.run_until_complete(oai_client.get_stream("What is the capital of France?"))
    # for msg in oai_client.chat.messages:
    #     print(msg)

    # Anthropic Example
    # print("\nAnthropic Example Response:")
    # anth_client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="anthropic")
    # loop = asyncio.get_event_loop()
    # response = loop.run_until_complete(anth_client.get_response("What is the capital of France?"))
    # for msg in anth_client.chat.messages:
    #     print(msg)

    # Ollama Example
    # print(f"\nOllama Example Response running {ollama_default_model}:")
    # ollama_client = AIChatAdapter(system_prompt="Answer as best as you can.", backend="ollama")
    # loop = asyncio.get_event_loop()
    # response = loop.run_until_complete(ollama_client.get_response("What is the capital of France?"))
    # for msg in ollama_client.chat.messages:
    #     print(msg)