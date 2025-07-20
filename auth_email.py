import smtplib
from email.mime.text import MIMEText
import os
import logging

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_USER = os.getenv("EMAIL_USER")             # e.g. daydreamforge@gmail.com
EMAIL_PASS = os.getenv("EMAIL_APP_PASSWORD")     # your 16-char Gmail app password

logger = logging.getLogger(__name__)

def send_login_code(to_email: str, code: str):
    if not EMAIL_USER or not EMAIL_PASS:
        logger.error("Missing EMAIL_USER or EMAIL_APP_PASSWORD env variable.")
        raise ValueError("Email credentials not set in environment variables.")
    logger.info(f"[send_login_code] Preparing to send code to {to_email}")

    body = f"Your DayDream Forge login code is: {code}"
    msg = MIMEText(body)
    msg["Subject"] = "Your DayDream Forge Login Code"
    msg["From"] = EMAIL_USER
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            logger.info("[send_login_code] Connecting to SMTP server...")
            server.login(EMAIL_USER, EMAIL_PASS)
            logger.info("[send_login_code] Logged in to SMTP server.")
            server.sendmail(msg["From"], [msg["To"]], msg.as_string())
            logger.info(f"[send_login_code] Email sent successfully to {to_email}.")
        # Do not return anything! Just complete if successful
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check your app password and email address.")
        raise
    except Exception as e:
        logger.exception(f"Failed to send login code email to {to_email}: {e}")
        raise
