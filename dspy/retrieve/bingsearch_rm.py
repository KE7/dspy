"""
Retriever model for Bing Search Engine
Author: Karim Elmaaroufi & Shangyin Tan
"""

from enum import Enum
from typing import List, Optional, Union

import requests
from dsp.utils.utils import dotdict
import dspy


class BingResultChoice(Enum):
    SNIPPET = "snippet"
    WEBPAGE = "webpage"


class BingSearch(dspy.Retrieve):

    def __init__(
            self, 
            bing_search_v7_subscription_key: str,
            bing_search_v7_endpoint: str,
            bing_result_choice: BingResultChoice,
            search_market: str = "en-us",
            k: int = 3,
        ):
        self.bing_search_v7_subscription_key = bing_search_v7_subscription_key
        self.bing_search_v7_endpoint = bing_search_v7_endpoint
        if "/v7.0/search" not in self.bing_search_v7_endpoint:
            self.bing_search_v7_endpoint += "/v7.0/search"
        self.bing_result_choice = bing_result_choice
        self.search_market = search_market
        self.k = k

        super().__init__(k=k)


    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None) -> dspy.Prediction:
        """Search Bing for self.k top passages for query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages = []

        for query in queries:
            results = self._search(query, k)

            passages.extend(dotdict({"long_text": passage}) for passage in results)

        return dspy.Prediction(
            passages=passages
        )
    

    def _search(self, query: str, k: int) -> List[str]:
        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_search_v7_subscription_key,
        }
        params = {
            "q": query, 
            "count": k, 
            "mkt": self.search_market,
        }
        try :
            response = requests.get(self.bing_search_v7_endpoint, headers=headers, params=params)
        except Exception as e:
            print("Error in Bing Search:\n", e)
            raise e

        response_json = response.json()
        if "webPages" not in response_json:
            # no search results came back
            # TODO: can we print a message or do something else here?
            return []
        
        if self.bing_result_choice == BingResultChoice.WEBPAGE:
            # we need to go get each webpage and return the result as a passage
            urls = [
                response_json["webPages"]["value"][i]["displayUrl"]
                for i in range(k)
            ]

            pages = [requests.get(url).text for url in urls]
            try:
                from bs4 import BeautifulSoup
                pages = [BeautifulSoup(page, "html.parser").get_text() for page in pages]
            except ImportError:
                raise ImportError(
                    "The 'beautifulsoup4' extra is required to use BingSearchRM with the 'webpage' result choice. Install it with `pip install dspy-ai[bingsearch]`",
                )
            
            # TODO: this is just raw html, need to do something better with it

            return pages
        else: # BingResultChoice.SNIPPET
            snippets = [
                response_json["webPages"]["value"][i]["snippet"]
                for i in range(k)
            ]

            return snippets
