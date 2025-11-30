from typing import cast
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.services.llm_service import FunctionCallParams


async def get_quote_of_the_day(params: FunctionCallParams):
    """
    Fetches from a knowledge base a very insightfull quote to present to the user.
    Share the output of this tool with the user in order to improve his day.
    """
    await params.result_callback(
        "Yesterday is past, Tomorrow is uncertain, today is all we have, this is why it is called a present"
    )


tools = ToolsSchema(standard_tools=[cast(DirectFunction, get_quote_of_the_day)])