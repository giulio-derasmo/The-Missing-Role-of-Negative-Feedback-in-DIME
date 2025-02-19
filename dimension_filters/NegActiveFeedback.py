import numpy as np
from .AbstractFilter import AbstractFilter


class NegActiveFeedback(AbstractFilter):
    """
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.hyperparams = kwargs["hyperparams"]

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)
        dlist = self.run[(self.run.query_id == query.query_id)].sort_values(by='relevance', ascending=False).doc_id.to_list()
        
        ### Relevant Feedback
        posemb = self.docs_encoder.get_encoding(dlist[0]) 
        relevant_feedback = np.multiply(qemb, posemb)
        ### Irrelevant Feedback
        negemb = self.docs_encoder.get_encoding(dlist[-1]) 
        irrelevant_feedback = np.multiply(qemb, negemb)

        ### Dimension Importance Score
        itx_vec =  self.hyperparams['alpha']*relevant_feedback - self.hyperparams['beta']*irrelevant_feedback
        return itx_vec