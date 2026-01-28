from abc import ABC, abstractmethod

from llama_index.core.tools import FunctionTool


class BaseTool(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def get_tools(self) -> list[FunctionTool]:
        pass


class ResultTool(BaseTool):
    def __init__(self):
        pass

    @abstractmethod
    async def refresh(self, result):
        pass
