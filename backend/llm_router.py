"""Provider-switchable LLM adapter for OpenAI, Gemini and Ollama."""
import requests

DEFAULTS = {
 'openai': {'model':'gpt-4.1-mini','base_url':'https://api.openai.com/v1'},
 'gemini': {'model':'gemini-2.5-flash','base_url':'https://generativelanguage.googleapis.com/v1beta/openai'},
 'ollama': {'model':'qwen2.5:7b','base_url':'http://localhost:11434'},
}

def chat(provider, model, api_key, base_url, messages):
    provider=(provider or 'ollama').lower(); cfg=DEFAULTS.get(provider)
    if not cfg: raise ValueError('Provider must be openai, gemini, or ollama')
    model=model or cfg['model']; base=(base_url or cfg['base_url']).rstrip('/')
    if provider == 'ollama':
        url=base if base.endswith('/api/chat') else base+'/api/chat'
        response=requests.post(url,json={'model':model,'messages':messages,'stream':False},timeout=120)
        response.raise_for_status(); return response.json()['message']['content']
    if not api_key: raise ValueError(f'Add an API key for {provider} in Settings.')
    # Gemini's OpenAI-compatible endpoint keeps the same provider adapter interface.
    url=base if base.endswith('/chat/completions') else base+'/chat/completions'
    response=requests.post(url,headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'},json={'model':model,'messages':messages,'temperature':0.35},timeout=120)
    response.raise_for_status(); return response.json()['choices'][0]['message']['content']
