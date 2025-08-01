#%% Transfer files with Globus
import globus_sdk
import webbrowser # To open the auth URL
import time     # For polling task status

# --- Configuration ---
# 1. Get your Client ID from developers.globus.org
CLIENT_ID = 'd4e66eb3-7ef2-4b1b-b580-3077e111d181' # Replace with your actual Client ID

# 2. Define Endpoint IDs and Paths
# Replace with your actual source endpoint ID
SOURCE_ENDPOINT_ID = 'f83d1248-5f5b-11ec-bded-55fe55c2cfea'
# Replace with your actual destination endpoint ID
DESTINATION_ENDPOINT_ID = '6df312ab-ad7c-4bbc-9369-450c82f0cb92'

# Define paths on the source and destination endpoints
# Example: Transfer a single file
SOURCE_PATH = '/Y/coskun-lab/Nicky/07 Temp/CopyPaste.txt'
DESTINATION_PATH = '/storage/home/hcoda1/5/nzhang326/p-acoskun7-0/Scripts/CopyPaste.txt'

# Example: Transfer a whole directory (recursively)
# SOURCE_PATH = '/path/on/source/endpoint/my_directory/' # Note the trailing slash
# DESTINATION_PATH = '/path/on/destination/endpoint/my_copied_directory/' # Note the trailing slash

# --- Authentication ---
# Use the Native App flow for interactive scripts
client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

# Define the required scopes for Transfer
# This typically includes 'openid', 'profile', 'email', and transfer scopes
requested_scopes = [
    'openid', 'profile', 'email',
    'urn:globus:auth:scope:transfer.api.globus.org:all'
    # Add scopes for specific endpoints if needed, e.g.,
    # f'urn:globus:auth:scope:endpoint:{SOURCE_ENDPOINT_ID}:all',
    # f'urn:globus:auth:scope:endpoint:{DESTINATION_ENDPOINT_ID}:all'
]

client.oauth2_start_flow(requested_scopes=requested_scopes, refresh_tokens=True)

authorize_url = client.oauth2_get_authorize_url()
print(f'Please go to this URL and login:\n\n{authorize_url}\n')

# Automatically open the URL in the default browser
try:
    webbrowser.open(authorize_url, new=1)
except webbrowser.Error:
    print("Could not open browser automatically.")

auth_code = input('Please enter the code you get after login here: ').strip()

try:
    token_response = client.oauth2_exchange_code_for_tokens(auth_code)
except globus_sdk.AuthAPIError as e:
    print(f"Authentication failed: {e.message} (HTTP {e.http_status})")
    exit(1)

globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']

# Extract the access token and refresh token
transfer_access_token = globus_transfer_data['access_token']
transfer_refresh_token = globus_transfer_data['refresh_token']
transfer_expires_at_s = globus_transfer_data['expires_at_seconds']

# --- Create Transfer Client ---
# Use the access token to create an authorizer
authorizer = globus_sdk.AccessTokenAuthorizer(transfer_access_token)
transfer_client = globus_sdk.TransferClient(authorizer=authorizer)

# --- Check Endpoint Activation (Optional but Recommended) ---
# Endpoints may require activation, especially Globus Connect Personal
print("Checking endpoint activation requirements...")
try:
    source_activation = transfer_client.endpoint_autoactivate(SOURCE_ENDPOINT_ID, if_expires_in=3600) # Activate for 1 hour
    dest_activation = transfer_client.endpoint_autoactivate(DESTINATION_ENDPOINT_ID, if_expires_in=3600)

    if source_activation["code"] == "AutoActivationFailed":
        print(f"Source Endpoint [{SOURCE_ENDPOINT_ID}] requires manual activation: "
              f"Please use this URL: {source_activation['activation_requirements_url']}")
        exit(1)
    if dest_activation["code"] == "AutoActivationFailed":
        print(f"Destination Endpoint [{DESTINATION_ENDPOINT_ID}] requires manual activation: "
              f"Please use this URL: {dest_activation['activation_requirements_url']}")
        exit(1)
    print("Endpoints activated or do not require activation.")

except globus_sdk.TransferAPIError as e:
    print(f"Error during endpoint activation check: {e.message} (HTTP {e.http_status})")
    # Decide if you want to proceed or exit
    # exit(1)

# --- Create TransferData Object ---
# Give the transfer a label (optional but helpful)
label = f"Python script transfer example - {time.strftime('%Y-%m-%d %H:%M:%S')}"
tdata = globus_sdk.TransferData(transfer_client,
                              source_endpoint=SOURCE_ENDPOINT_ID,
                              destination_endpoint=DESTINATION_ENDPOINT_ID,
                              label=label,
                              sync_level=None) # sync_level can be 'checksum', 'size', 'mtime'

# --- Add Items to Transfer ---
# Check if source path ends with '/' for directory transfer
is_recursive = SOURCE_PATH.endswith('/')

tdata.add_item(source_path=SOURCE_PATH,
               destination_path=DESTINATION_PATH,
               recursive=is_recursive) # Set recursive=True for directories

# Add more items if needed:
# tdata.add_item('/path/on/source/another_file.dat', '/path/on/dest/another_file.dat')

# --- Submit the Transfer Task ---
try:
    submission_result = transfer_client.submit_transfer(tdata)
    task_id = submission_result['task_id']
    print(f"Transfer task submitted successfully!")
    print(f"Task ID: {task_id}")
    print(f"You can monitor the task here: https://app.globus.org/activity/{task_id}")
except globus_sdk.TransferAPIError as e:
    print(f"Error submitting transfer: {e.message} (HTTP {e.http_status})")
    exit(1)

# --- Monitor the Transfer Task (Optional) ---
print("Monitoring task status (checking every 60 seconds)...")
while True:
    try:
        task = transfer_client.get_task(task_id)
        status = task['status']
        if status == 'SUCCEEDED':
            print("Task completed successfully!")
            break
        elif status == 'FAILED':
            print(f"Task failed! Reason: {task.get('fatal_error', {}).get('description', 'Unknown')}")
            break
        elif status == 'ACTIVE':
            print(f"Task is active (progress: {task.get('bytes_transferred', 0)} bytes)...")
        elif status == 'INACTIVE':
             print(f"Task is inactive (e.g., waiting for activation)...")
        else:
             print(f"Task status: {status}")

        # Avoid busy-waiting
        time.sleep(60) # Check every 60 seconds

    except globus_sdk.TransferAPIError as e:
        print(f"Error checking task status: {e.message}. Will retry.")
        time.sleep(60) # Wait before retrying
    except KeyboardInterrupt:
         print("\nMonitoring interrupted by user.")
         break

print("Script finished.")