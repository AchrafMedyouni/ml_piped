import traceback
import re
from time import sleep
import mlflow
from google import genai

# ─── Configuration Gemini CLIENT ───────────────────────────────────────────
API_KEY = "AIzaSyDWklovIvU6F6n3xUqQiqIvpDVTmx53zdc"  # Remplacez par votre clé en clair
client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.0-flash"

def try_run_pipeline(code):
    """Tries running the pipeline and returns (None, None) if all is well, or (error, traceback) otherwise."""
    try:
        exec(code, {})
        return None, None
    except Exception as e:
        return str(e), traceback.format_exc()


def ask_gemini_to_fix(code: str, error: str, tb: str) -> str:
    """Envoie le code et l'erreur à Gemini, renvoie la réponse brute."""
    prompt = f"""
This Python ML pipeline throws a runtime error.

❌ Error:
{error}

🔍 Traceback:
{tb}

💻 Full code:
```python
{code}
✅ Please provide a corrected, complete version of the code within a single python block.
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text

def extract_code(reply: str) -> str | None:
    """Extracts Python code contained in a Markdown ```python block``` if present."""
    match = re.search(r"```python\n(.+?)```", reply, re.S)
    return match.group(1) if match else None    
def main(filepath: str = "pipeline.py"):
    """Main function to run the pipeline and fix it if it fails."""
    orig_file = filepath
    fixed_file = "pipeline_fixed.py"

    with open(orig_file, "r", encoding="utf-8") as f:
        code = f.read() 
    # Try running the pipeline
    try:
        error, tb = try_run_pipeline(orig_file)
    except Exception as e:
        print("⚠️ Error while trying to run the pipeline:", e)
        
    if error is None:
        print("✅ Pipeline ran successfully—no fix needed.")
        return      
    print("❌ Pipeline failed. Requesting fix from Gemini…")
    current_code = code
    current_error = error
    current_tb = tb
    i = 0

    while current_error is not None:
        fix_reply = ask_gemini_to_fix(current_code, current_error, current_tb)
        fixed_code = extract_code(fix_reply)
        if fixed_code is None:
            print("⚠️ Couldn't parse the fixed code. Here’s the full Gemini reply:\n")
            print(fix_reply)
            return
        #running the fixed code
        current_code = fixed_code
        current_error, current_tb = try_run_pipeline(current_code)
        i += 1
        if current_error is None:
            print(f"✅ Pipeline fixed successfully after {i} iterations.")
            with open(fixed_file, "w", encoding="utf-8") as f:
                f.write(current_code)
            return
        else:
            print(f"❌ Fix attempt {i} failed with error: {current_error}")
        if i >= 5:
            print("⚠️ Too many iterations without success. Stopping.")
            return
        
if __name__ == "__main__":
    main()