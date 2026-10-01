import os
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("INRIX_APP_ID")
APP_KEY = os.getenv("INRIX_APP_KEY")

if not APP_ID or not APP_KEY:
    raise RuntimeError("INRIX_APP_ID or INRIX_APP_KEY is missing from .env")

# INRIX authentication:
# lowercase(AppId|AppKey) -> UTF-8 -> SHA1 -> hexadecimal
hash_input = f"{APP_ID}|{APP_KEY}".lower()
hash_token = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()

response = requests.get(
    "https://uas-api.inrix.com/v1/AppToken",
    params={
        "appId": APP_ID,
        "hashToken": hash_token,
    },
    headers={
        "Accept": "application/json",
    },
    timeout=20,
)

print("HTTP STATUS:", response.status_code)

try:
    data = response.json()
except Exception:
    print("Invalid JSON response")
    raise SystemExit(1)

if response.ok:
    result = data.get("result", {})

    print("INRIX AUTHENTICATION: PASS")
    print("Token received:", bool(result.get("token")))
    print("Token expiry:", result.get("expiry"))
else:
    error = data.get("error", {})
    print("INRIX AUTHENTICATION: FAILED")
    print("Status ID:", error.get("statusId"))
    print("Message:", error.get("userMessage"))