import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from typing import List
from pydantic import BaseModel

class ResearchState(BaseModel):
    query: str
    literature_results: str
    summary_results: str
    hypothesis_results: str
    errors: List[str]
    final_output: str
    
