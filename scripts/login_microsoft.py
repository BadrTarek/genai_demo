from etic_genai.tools.document_access.microsoft_auth import finish_device_login, start_device_login
"""a standalone helper script for testing Microsoft device sign-in locally:"""

if __name__ == "__main__":
    user_id = "local-dev"
    print("Starting Microsoft device login...")
    start_response = start_device_login(user_id)
    print("Start Response:", start_response["message"])

    input("Press Enter after completing the device login...")

    print("Finishing Microsoft device login...")
    finish_response = finish_device_login(user_id)
    print(f"signed in as {finish_response['account']} for user_id '{user_id}'")