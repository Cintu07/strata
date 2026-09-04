"""m0: the measurement harness that decides whether strata is worth building.

the project's falsification test. it instruments a real mixture of experts model
and answers five questions, any one of which can end the project:

1. do consecutive tokens reuse experts, or is routing effectively random
2. how skewed is router load
3. what hit rate is achievable at a given ram budget
4. do experts fire together in stable groups
5. can layer L+k routing be predicted from layer L's hidden state

see ``report.run`` for the verdict, and ``synthetic.make_trace`` for the traces
used to test the harness itself.
"""

from .trace import Provenance, RouterTrace, SOURCE_CAPTURED, SOURCE_SYNTHETIC

__all__ = ["Provenance", "RouterTrace", "SOURCE_CAPTURED", "SOURCE_SYNTHETIC"]
__version__ = "0.1.0"
