import requests

# Replace 'yourshortusername' with your actual short username
shortname = ""
url = "http://cs7ns1.scss.tcd.ie"
params = {"shortname": shortname}

# Make the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    print("Success!")
    # Save the response content to a text file
    with open("response_output.txt", "w") as file:
        file.write(response.text)
    print("Response saved to response_output.txt")
else:
    print(f"Failed with status code: {response.status_code}")
