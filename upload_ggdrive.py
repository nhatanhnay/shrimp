from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SERVICE_ACCOUNT_FILE = 'key.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = '1He0GAdbkKpgXES_YOsH3bfMY0_Swjw8R'  # ID thư mục Drive bạn đã chia sẻ

def upload_to_drive(file_path, file_name):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': file_name,
        'parents': [FOLDER_ID]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(
        body=file_metadata, media_body=media, fields='id, webViewLink'
    ).execute()

    print(f"Done")
if __name__ == '__main__':
# Gọi hàm:
    upload_to_drive('tomtep.mp4', 'test1.mp4')
