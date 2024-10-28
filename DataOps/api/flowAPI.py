# URL - https://app.prefect.cloud/account/8ff8f613-92c4-44ce-b811-f9956023e78d/workspace/04d8fca9-df2e-40c8-ae4f-a3733114c475/dashboard

# URL - https://app.prefect.cloud/api/docs

import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "pnu_gBrxFQP6DZh9fSY9WBRmIur4yC3Yy928ggGT"  # Your Prefect Cloud API key
ACCOUNT_ID = "cc92ab2b-70f8-4de7-9972-945978bc51d9"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "2e49cb16-5112-408d-a48f-68661136c7fa"  # Your Prefect Cloud Workspace ID
FLOW_ID = "8bc0051d-7a89-4002-ad53-2b9bb750e54a"  # Your Flow ID

# Correct API URL to get flow details
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/flows/{FLOW_ID}"

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}

# Make the request using GET
response = requests.get(PREFECT_API_URL, headers=headers)

# Check the response status
if response.status_code == 200:
    flow_info = response.json()
    print(flow_info)
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
