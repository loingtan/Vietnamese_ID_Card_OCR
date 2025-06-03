#!/usr/bin/env python3
"""
Start FastAPI app với Slack webhook URL configured
"""

import os
import sys

# Set environment variables
os.environ["SLACK_WEBHOOK_URL"] = "https://hooks.slack.com/services/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"
os.environ["TELEGRAM_BOT_TOKEN"] = ""  # Add if needed
os.environ["TELEGRAM_CHAT_ID"] = ""    # Add if needed

# Add src to path
sys.path.append('src')

if __name__ == "__main__":
    from src.api.fastapi_app import main
    main()
