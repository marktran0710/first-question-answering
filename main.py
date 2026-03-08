import json
import os
import sys
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

# Load .env from the same directory as this script
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HISTORY_FILE = "chat_history.json"

# ==========================================
# 1. Define the Python Functions
# ==========================================
def get_exchange_rate(currency_pair: str) -> str:
    """Mock function to get the exchange rate."""
    data = {"USD_TWD": "32.0", "JPY_TWD": "0.2", "EUR_USD": "1.2"}
    if currency_pair in data:
        return json.dumps({"currency_pair": currency_pair, "rate": data[currency_pair]})
    return json.dumps({"error": "Data not found"})

def get_stock_price(symbol: str) -> str:
    """Mock function to get the stock price."""
    data = {"AAPL": "260.00", "TSLA": "430.00", "NVDA": "190.00"}
    if symbol in data:
        return json.dumps({"symbol": symbol, "price": data[symbol]})
    return json.dumps({"error": "Data not found"})

# ==========================================
# 2. Function Map (Dispatch Dictionary)
# ==========================================
available_functions = {
    "get_exchange_rate": get_exchange_rate,
    "get_stock_price": get_stock_price
}

# ==========================================
# 3. Tool Schemas (Groq / Standard OpenAI Format)
# ==========================================
tools =[
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the exchange rate for a given currency pair.",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency_pair": {
                        "type": "string",
                        "description": "The currency pair (e.g., 'USD_TWD')"
                    }
                },
                "required": ["currency_pair"],
                "strict": True,
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock symbol (e.g., 'AAPL')"
                    }
                },
                "required":["symbol"],
                "strict": True,
                "additionalProperties": False
            }
        }
    }
]

# ==========================================
# 4. History Persistence Helpers
# ==========================================
# Groq/OpenAI history is just a standard Python list of dictionaries, 
# making serialization incredibly easy!

def save_history(messages: list):
    """Save chat history directly to a JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"[System: History saved to {HISTORY_FILE} ({len(messages)} turns)]")

def load_history(system_instruction: str) -> list:
    """Load chat history, or create a new file with system prompt."""

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            messages = json.load(f)

        print(f"[System: Loaded {len(messages)} turns from {HISTORY_FILE}]")
        return messages

    # If history file does not exist → create it
    messages = [{"role": "system", "content": system_instruction}]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    print(f"[System: Created new history file {HISTORY_FILE}]")

    return messages

def clear_history(system_instruction: str) -> list:
    """Delete the history file and return a fresh messages list."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        print(f"[System: History cleared.]")
    else:
        print("[System: No history file found.]")
    return [{"role": "system", "content": system_instruction}]


# ==========================================
# 5. Main Agent Loop
# ==========================================
def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found in .env file!")
        sys.exit(1)

    client = Groq(api_key=api_key)

    system_instruction = (
        "You are a Financial Assistant specializing in stock prices and exchange rates. "
        "Always use tools to fetch data. For multiple stocks, call all tools in parallel. "
        "After fetching, compare prices clearly: show each value, the difference, and which is higher. "
        "On tool error, inform the user gracefully. "
        "Remember previous tool results across the conversation."
    )

    # Load history (or initialize it)
    messages = load_history(system_instruction)

    print("\nWelcome to the Financial CLI Chatbot (Powered by Groq)!")
    print("Try asking: 'What is the price of AAPL and TSLA?' or 'Convert USD to TWD'")
    print("Commands: 'exit'/'quit' to stop, 'clear history' to reset memory.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            save_history(messages)
            print("Goodbye!")
            break

        if user_input.lower() == "clear history":
            messages = clear_history(system_instruction)
            print("[System: Memory cleared. Starting fresh.]\n")
            continue

        if not user_input:
            continue

        # 1. Add user input to memory
        messages.append({"role": "user", "content": user_input})

        # Robust loop: keeps running if the model requests parallel tools
        while True:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Groq's best tool-calling model right now
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0
            )

            response_message = response.choices[0].message

            # 2. Append the assistant's response to memory FIRST.
            # Convert the Pydantic object into a clean dictionary so it can be saved to JSON easily later
            messages.append(response_message.model_dump(exclude_unset=True))

            # 3. Check if the model called any tools
            if not response_message.tool_calls:
                # If no tools were called, print the final answer and break the inner loop
                print(f"\nAssistant: {response_message.content}\n")
                break

            # 4. Handle tool calls (Parallel Execution)
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args_str = tool_call.function.arguments # This comes back as a JSON string

                print(f"[System: Executing {func_name}(**{func_args_str})]...")

                func_to_call = available_functions.get(func_name)
                
                if func_to_call:
                    try:
                        # Parse the JSON arguments to a Python dictionary
                        args_dict = json.loads(func_args_str)
                        # Execute function
                        result_str = func_to_call(**args_dict)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                else:
                    result_str = json.dumps({"error": f"Function {func_name} not found"})

                # 5. Append the result of the tool execution back into memory
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result_str
                })
            
            # The loop continues to send the tool results back to the LLM!

        # Auto-save history after every complete turn
        save_history(messages)

if __name__ == "__main__":
    main()