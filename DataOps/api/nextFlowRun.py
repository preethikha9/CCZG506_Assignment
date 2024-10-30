import requests
import json

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "pnu_gBrxFQP6DZh9fSY9WBRmIur4yC3Yy928ggGT"  # Your Prefect Cloud API key
ACCOUNT_ID = "cc92ab2b-70f8-4de7-9972-945978bc51d9"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "2e49cb16-5112-408d-a48f-68661136c7fa"  # Your Prefect Cloud Workspace ID

# Correct API URL to get deployment details
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/ui/flows/next-runs"

# Set up headers with Authorization
headers = {
    "Authorization": f"Bearer {PREFECT_API_KEY}",
    "Content-Type": "application/json"
}

# Define the request body
data = {
    "flow_ids": ["5578d338-82a1-444d-acb2-78fa5f62c554"]
}

# Make the request using POST
response = requests.post(PREFECT_API_URL, headers=headers, data=json.dumps(data))

# Check the response status
if response.status_code == 200:
    deployment_info = response.json()
    print(json.dumps(deployment_info, indent=2))
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
