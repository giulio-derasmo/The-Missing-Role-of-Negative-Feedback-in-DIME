import numpy as np
import pandas as pd

from .AbstractFilter import AbstractFilter


class GPTFilter(AbstractFilter):
    """
    empty filter which returns all the dimensions
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)
        self.model = kwargs["model"]
        self.gpt_answers = pd.read_csv(kwargs["answers_path"], dtype={"query_id": str})
        self.gpt_answers = self.gpt_answers.loc[~self.gpt_answers["query_id"].duplicated()].set_index("query_id")

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)

        if query.query_id in self.gpt_answers.index:
            aemb = self.model.encode(self.gpt_answers.loc[query.query_id, "text"])
        else:
            print(f"query {query.query_id} gpt answer not available")
            aemb = qemb

        itx_vec = np.multiply(qemb, aemb)

        return itx_vec