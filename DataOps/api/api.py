# URL - https://app.prefect.cloud/account/8ff8f613-92c4-44ce-b811-f9956023e78d/workspace/04d8fca9-df2e-40c8-ae4f-a3733114c475/dashboard

# Ref - https://app.prefect.cloud/api/docs

import requests

# Replace these variables with your actual Prefect Cloud credentials
PREFECT_API_KEY = "pnu_gBrxFQP6DZh9fSY9WBRmIur4yC3Yy928ggGT"  # Your Prefect Cloud API key
ACCOUNT_ID = "cc92ab2b-70f8-4de7-9972-945978bc51d9"  # Your Prefect Cloud Account ID
WORKSPACE_ID = "2e49cb16-5112-408d-a48f-68661136c7fa"  # Your Prefect Cloud Workspace ID

# Correct API URL to list flow runs
PREFECT_API_URL = f"https://api.prefect.cloud/api/accounts/{ACCOUNT_ID}/workspaces/{WORKSPACE_ID}/artifacts/filter"

# Data to filter artifacts
data = {
    "sort": "CREATED_DESC",
    "limit": 5,
    "artifacts": {
        "key": {
            "exists_": True
        }
    }
}

# Set up headers with Authorization
headers = {"Authorization": f"Bearer {PREFECT_API_KEY}"}

# Make the request
response = requests.post(PREFECT_API_URL, headers=headers, json=data)
print(response)

# Check the response status
if response.status_code != 200:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response content: {response.text}")
else:
    artifacts = response.json()
    print(artifacts)
    for artifact in artifacts:
        print(artifact)
