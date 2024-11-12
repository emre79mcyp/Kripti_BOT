import hmac
import hashlib
import time
import requests

API_KEY = 'xphI8ywTK6IIyyLXhvoSLF8MEWbGPbkC0UFtp7wBRrDvp8UkJMtNdl5ifj28L2Lp'
SECRET_KEY = 'ykGSrhFx79Wb5tFZ8oJaZ15Xqb7Kk65IaxZ4so7K6IkWWGvfKw8Hjoc9J1wJ6PGh'
BASE_URL = 'https://api.binance.com/api/v3'

# Function to create signature
def sign_request(secret_key, params):
    query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

# Function to test API key validation
def test_validate_api_keys():
    try:
        # Log API Key length and value for debugging (not for production use)
        print(f"API Key Length: {len(api_key)}")
        print(f"API Key: {api_key}")
        
        # Define the API endpoint
        endpoint = f"{base_url}/account"
        
        # Generate the timestamp and signature
        timestamp = int(time.time() * 1000)
        params = {'timestamp': timestamp}
        params['signature'] = sign_request(secret_key, params)
        
        # Define the headers with the API key
        headers = {'X-MBX-APIKEY': api_key}
        
        # Make the API request
        response = requests.get(endpoint, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Print the successful response
        print(response.json())
    except requests.exceptions.HTTPError as err:
        # Print the error message and API response for debugging
        print(f"Error: {err}")
        print(response.text)

# Call the function to test API key validation
test_validate_api_keys()