from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
import time
from twilio.rest import Client

# Load model
model = load_model('lstm_ecg_best.keras')

# ThingSpeak and Twilio configuration
WRITE_API_KEY = 'YOUR_WRITE_API_KEY'
READ_API_KEY = 'YOUR_READ_API_KEY'
CHANNEL_ID = 'YOUR_CHANNEL_ID'
ECG_FIELD_NUMBER = 5
TRIGGER_FIELD_NUMBER = 6
ALERT_FIELD_NUMBER = 6

account_sid = 'YOUR_TWILIO_SID'
auth_token = 'YOUR_TWILIO_TOKEN'
twilio_number = 'YOUR_TWILIO_PHONE'
recipient_number = 'YOUR_PHONE_NUMBER'

def update_alert_gauge(value):
    requests.post("https://api.thingspeak.com/update", data={
        'api_key': WRITE_API_KEY,
        f'field{ALERT_FIELD_NUMBER}': value
    })

def fetch_ecg_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/fields/{ECG_FIELD_NUMBER}.json?results=1&api_key={READ_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            feeds = response.json()["feeds"]
            if feeds and feeds[0].get(f'field{ECG_FIELD_NUMBER}') is not None:
                return float(feeds[0][f'field{ECG_FIELD_NUMBER}'])
        return None
    except:
        return None

def fetch_trigger_status():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/fields/{TRIGGER_FIELD_NUMBER}.json?results=1&api_key={READ_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            feeds = response.json()["feeds"]
            if feeds and feeds[0].get(f'field{TRIGGER_FIELD_NUMBER}') is not None:
                return int(feeds[0][f'field{TRIGGER_FIELD_NUMBER}'])
        return 0
    except:
        return 0

def send_sms_alert(score):
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=f"🚨 ECG Alert: Abnormal heartbeat detected! (Score: {score:.4f})",
        from_=twilio_number,
        to=recipient_number
    )
    print("SMS sent:", message.sid)

ecg_values = []
while True:
    if fetch_trigger_status() == 1:
        ecg_val = fetch_ecg_data()
        if ecg_val is not None:
            ecg_values.append(ecg_val)
            if len(ecg_values) >= 60:
                scaler = StandardScaler()
                ecg_scaled = scaler.fit_transform(np.array(ecg_values).reshape(-1, 1)).flatten()
                ecg_input = ecg_scaled.reshape(1, len(ecg_scaled), 1)
                output = model.predict(ecg_input)
                pred_class = np.argmax(output, axis=1)[0]
                print("Prediction:", pred_class, "Probabilities:", output[0])
                if pred_class > 0:
                    update_alert_gauge(1)
                    send_sms_alert(output[0][0])
                else:
                    update_alert_gauge(0)
                ecg_values = []
    else:
        print("Trigger off. Waiting...")
        ecg_values = []
    time.sleep(2)
