import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
def loadapi():
    from dotenv import load_dotenv
    import os
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")