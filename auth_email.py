# auth_email.py

import smtplib
from email.mime.text import MIMEText
import os

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_USER = os.getenv("EMAIL_USER")             # e.g. daydreamforge@gmail.com
EMAIL_PASS = os.getenv("EMAIL_APP_PASSWORD")     # your 16-char Gmail app password

def send_login_code(to_email: str, code: str):
    body = f"Your DayDream Forge login code is: {code}"

    msg = MIMEText(body)
    msg["Subject"] = "Your DayDream Forge Login Code"
    msg["From"] = EMAIL_USER
    msg["To"] = to_email

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(msg["From"], [msg["To"]], msg.as_string())
