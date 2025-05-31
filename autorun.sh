# ~/run_main.sh
#!/usr/bin/env bash
# Script này đảm bảo Python luôn chạy trong thư mục dự án
# --- chỉnh đường dẫn dưới cho đúng ---
PROJECT_DIR="/home/your_user/ProjectFinal/VSLR"
PYTHON="/usr/bin/python3"        # Hoặc đường dẫn khác tới python env

cd "$PROJECT_DIR" || exit 1
exec $PYTHON main.py >> "$PROJECT_DIR/log_main.out" 2>&1
