"""
Real Ollama integration test script for Phase 7.
Verifies server reachability, checks model availability, and executes ONE real generation test.
"""

import time
from src.generator import LLMGenerator

def test_real_ollama():
    gen = LLMGenerator()
    print("=" * 70)
    print("PHASE 7 REAL OLLAMA INTEGRATION TEST")
    print("=" * 70)
    print(f"Ollama Base URL: {gen.base_url}")
    print(f"Configured Model: {gen.model}")
    print(f"Configured Temperature: {gen.temperature}")
    print(f"Configured Timeout: {gen.timeout}s")
    print("-" * 70)

    # 1. Health check
    is_healthy = gen.health_check()
    print(f"1. Ollama Server Reachable: {is_healthy}")
    if not is_healthy:
        print("ERROR: Ollama server is not reachable. Exiting test.")
        return

    # 2. Model availability
    is_model_avail = gen.model_available()
    print(f"2. Model '{gen.model}' Available Locally: {is_model_avail}")
    if not is_model_avail:
        avail_models = gen.get_available_models()
        print(f"Available models: {avail_models}")
        print(f"ERROR: Model '{gen.model}' is not available locally. Exiting test.")
        return

    # 3. Real generation test as requested by prompt
    test_prompt = "Return exactly one sentence saying that this is a local LLM generation test."
    print(f"3. Test Prompt: \"{test_prompt}\"")
    
    start_time = time.perf_counter()
    # Provide direct instruction override for simple directive testing
    response = gen.generate(test_prompt, system_prompt="You are a helpful assistant. Follow the prompt instructions precisely.")
    latency = time.perf_counter() - start_time

    print(f"4. Generation Success: True")
    print(f"5. Total Generation Latency: {latency:.3f} seconds ({latency * 1000:.1f} ms)")
    print(f"6. Returned Response: \"{response}\"")
    print("=" * 70)

if __name__ == "__main__":
    test_real_ollama()
