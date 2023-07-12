在agent/bing_search.py下：

```python
#coding=utf8

from langchain.utilities import BingSearchAPIWrapper
from configs.model_config import BING_SEARCH_URL, BING_SUBSCRIPTION_KEY


def bing_search(text, result_len=3):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [{"snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                 "title": "env info is not found",
                 "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html"}]
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL)
    return search.results(text, result_len)


if __name__ == "__main__":
    r = bing_search('python')
    print(r)
```

BING_SUBSCRIPTION_KEY需要我们去申请，看看

langchain/utilities/bing_search.py

```python
"""Util that calls Bing Search.

In order to set this up, follow instructions at:
https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e
"""
from typing import Dict, List

import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env


class BingSearchAPIWrapper(BaseModel):
    """Wrapper for Bing Search API.

    In order to set this up, follow instructions at:
    https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e
    """

    bing_subscription_key: str
    bing_search_url: str
    k: int = 10

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _bing_search_results(self, search_term: str, count: int) -> List[dict]:
        headers = {"Ocp-Apim-Subscription-Key": self.bing_subscription_key}
        params = {
            "q": search_term,
            "count": count,
            "textDecorations": True,
            "textFormat": "HTML",
        }
        response = requests.get(
            self.bing_search_url, headers=headers, params=params  # type: ignore
        )
        response.raise_for_status()
        search_results = response.json()
        return search_results["webPages"]["value"]

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        bing_subscription_key = get_from_dict_or_env(
            values, "bing_subscription_key", "BING_SUBSCRIPTION_KEY"
        )
        values["bing_subscription_key"] = bing_subscription_key

        bing_search_url = get_from_dict_or_env(
            values,
            "bing_search_url",
            "BING_SEARCH_URL",
            # default="https://api.bing.microsoft.com/v7.0/search",
        )

        values["bing_search_url"] = bing_search_url

        return values

    def run(self, query: str) -> str:
        """Run query through BingSearch and parse result."""
        snippets = []
        results = self._bing_search_results(query, count=self.k)
        if len(results) == 0:
            return "No good Bing Search Result was found"
        for result in results:
            snippets.append(result["snippet"])

        return " ".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through BingSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        metadata_results = []
        results = self._bing_search_results(query, count=num_results)
        if len(results) == 0:
            return [{"Result": "No good Bing Search Result was found"}]
        for result in results:
            metadata_result = {
                "snippet": result["snippet"],
                "title": result["name"],
                "link": result["url"],
            }
            metadata_results.append(metadata_result)

        return metadata_results
```

实际上就是使用的bing的api接口，需要注意的是返回值是什么。
