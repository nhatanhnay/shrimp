from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Nhập các thư viện cần thiết từ Google API Client để xác thực và tải file lên Google Drive
# `service_account` dùng để xác thực bằng tài khoản dịch vụ
# `build` để tạo client API
# `MediaFileUpload` để tải file

SERVICE_ACCOUNT_FILE = 'key.json'  # Đường dẫn đến file JSON chứa thông tin xác thực tài khoản dịch vụ
SCOPES = ['https://www.googleapis.com/auth/drive']  # Phạm vi quyền truy cập, ở đây là quyền truy cập Google Drive
FOLDER_ID = '1He0GAdbkKpgXES_YOsH3bfMY0_Swjw8R'  # ID của thư mục trên Google Drive nơi file sẽ được tải lên

# Hàm tải file lên Google Drive, nhận đường dẫn file cục bộ và tên file trên Drive
def upload_to_drive(file_path, file_name):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)  # Tạo đối tượng thông tin xác thực từ file `key.json` với các quyền được chỉ định trong `SCOPES`
    
    service = build('drive', 'v3', credentials=creds)  # Tạo client API cho Google Drive (phiên bản v3) sử dụng thông tin xác thực

    file_metadata = {
        'name': file_name,
        'parents': [FOLDER_ID]
    }  # Tạo metadata cho file, bao gồm tên file (`name`) và ID thư mục cha (`parents`) để xác định nơi lưu file trên Drive
    
    media = MediaFileUpload(file_path, resumable=True)  # Tạo đối tượng `MediaFileUpload` từ đường dẫn file, bật chế độ tải có thể tiếp tục (`resumable=True`) để xử lý các lỗi mạng
    
    file = service.files().create(
        body=file_metadata, media_body=media, fields='id, webViewLink'
    ).execute()  # Gửi yêu cầu tạo file mới trên Google Drive, tải file lên và trả về `id` và `webViewLink` của file

    print(f"Done")  # In thông báo hoàn tất tải file

if __name__ == '__main__':
    # Thử nghiệm hàm `upload_to_drive` bằng cách tải file `tomtep.mp4` lên Drive với tên `test1.mp4`
    upload_to_drive('tomtep.mp4', 'test1.mp4')