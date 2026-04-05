import json
import urllib.request
import sys

# --- CONFIGURATION (Fill these in!) ---
# 1. Get this from Firebase Console -> Project Settings -> General -> "Web API Key"
API_KEY = "AIzaSyBJRlSTsRTXPFaOO_ErgKUOj5ILo8XJNdE"

# 2. Create a test user in Firebase Console -> Authentication -> Users, and put the credentials here:
EMAIL = "test@gmail.com"
PASSWORD = "test@123"
# ---------------------------------------

def get_token():
    print(f"Logging in to Firebase as {EMAIL}...")
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    data = json.dumps({
        "email": EMAIL,
        "password": PASSWORD,
        "returnSecureToken": True
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            print("\n✅ SUCCESS! Here is your Firebase ID token (it will expire in 1 hour):\n")
            print(result["idToken"])
            print("\n🔑  Next Steps:")
            print("1. Start your backend:  uvicorn main:app --reload")
            print("2. Open your browser to: http://127.0.0.1:8000/docs")
            print("3. Click the green 'Authorize' button at the top right.")
            print("4. Paste this token into the Value box and click Authorize.")
            print("5. Try testing the endpoints!")
            
    except urllib.error.HTTPError as e:
        error_msg = json.loads(e.read().decode('utf-8'))
        print(f"\n❌ Login Failed: {error_msg.get('error', {}).get('message', 'Unknown Error')}")
        print("Did you make sure to create the user in Firebase Auth and enable Email/Password login?")

if __name__ == "__main__":
    if API_KEY == "YOUR_WEB_API_KEY_HERE":
        print("⚠️  Wait! You need to configure the script first.")
        print("Open get_test_token.py and replace YOUR_WEB_API_KEY_HERE with your Firebase Web API Key.")
        sys.exit(1)
    get_token()
