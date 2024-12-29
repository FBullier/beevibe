from openai import OpenAI
import httpx
import configparser

from beevibe.utils.logger import setup_logger

from typing import Any

class AIHelper:

    class CustomHTTPClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("proxy", None)  # Remove the 'proxies' argument if present
            super().__init__(*args, **kwargs)


    def __init__(self, config_file: str = "config.ini", verbose: bool = True):
        self.verbose = verbose
        self.logger = self.__init_logger()

        # Load OpenAI key from .ini file
        openai_key = self.__load_openai_key(config_file)

        # Check if API key is loaded
        if openai_key:
            self.api_key = openai_key
            api_key = self.api_key

            self.logger_info("Found an API key")
        else:
            raise ValueError(f"API key not found in {config_file}")

        self.client = OpenAI(
            http_client=self.CustomHTTPClient(),
            api_key=api_key,
        )


    def __init_logger(self) -> Any:
        """
        Initialize the logger for the trainer.

        Returns:
            Any: The initialized logger.
        """
        return setup_logger()


    def logger_info(self, message: str) -> None:
        """
        Log a message if verbose logging is enabled.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            self.logger.info(message)


    def __load_openai_key(self, config_file: str) -> str:
        """
        Load the OpenAI key from a configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            str: The OpenAI API key.
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        return config.get("openai", "api_key", fallback=None)


    def help(self, question: str) -> str:
        """
        Get help about the Beevibe project.

        Args:
            question (str): The question to ask.

        Returns:
            str: The assistant's response.
        """
        messages = [
            {"role": "system", "content": "You are a friendly assistant helping developers understand a Python project."},
            {"role": "system", "content": "You answer questions about the Flair GitHub project you will find here: https://github.com/flairNLP/flair."},
            {"role": "system", "content": "Don't answer questions outside the scope of this project."},
            {"role": "user", "content": question},
        ]

        try:

            response = self.client.chat.completions.create(
                messages=messages,
                model="gpt-4o-2024-08-06",
            )            

            #response = await openai.ChatCompletion.create(
            #    model="gpt-4",
            #    messages=messages,
            #)
            
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"Error in OpenAI API call: {e}")
            return "An error occurred while fetching the response."

