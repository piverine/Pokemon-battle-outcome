import requests

API_KEY = "YOUR_API_KEY"  # Replace with your actual VirusTotal API key
FILE_PATH = "app.py"  # Change this to the file you want to upload

url = "https://www.virustotal.com/api/v3/files"

headers = {
    "accept": "application/json",
    "x-apikey": "2df3691da27fb12ace8c53116434876cf2193783bc776f900001ced057a6c924"
}

# Open the file in binary mode
with open("mergesort.py", "rb") as file:
    files = {"file": ("mergesort.py", file, "application/octet-stream")}
    response = requests.post(url, files=files, headers=headers)

# Print the API response
print(response.json())  # Print as JSON for better readability
print("Status Code:", response.status_code)
print("Response Headers:", response.headers)
print("Response Body:", response.text)
print("Scan Results:", response.json())

# import requests
# 
# API_KEY = "YOUR_API_KEY"  # Replace with your actual key
# url = "https://www.virustotal.com/api/v3/users/me"
# 
# headers = {
#     "accept": "application/json",
#     "x-apikey": "2df3691da27fb12ace8c53116434876cf2193783bc776f900001ced057a6c924"
# }
# 
# response = requests.get(url, headers=headers)
# print(response.json())
