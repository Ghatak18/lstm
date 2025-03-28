import socket
import time
import random
import json
from datetime import datetime

# Socket configuration
HOST = "localhost"
PORT = 5000

# Define user behavior based on time of day
USER_TYPES = {
    "Office Worker": {"min": 10, "max": 50, "active": (8, 17)},
    "Streamer": {"min": 50, "max": 100, "active": (17, 24)},
    "Gamer": {"min": 40, "max": 90, "active": (18, 2)},
    "Casual User": {"min": 5, "max": 20, "active": (0, 24)},
    "Night Worker": {"min": 20, "max": 50, "active": (22, 5)}
}

# Number of users to simulate
NUM_USERS = 10

def get_time_based_bandwidth(user_type):
    """Generate bandwidth usage based on user type and time of day."""
    current_hour = datetime.now().hour
    min_bw, max_bw = USER_TYPES[user_type]["min"], USER_TYPES[user_type]["max"]
    start, end = USER_TYPES[user_type]["active"]

    # Adjust based on active hours
    if start <= current_hour <= end or (start > end and (current_hour >= start or current_hour <= end)):
        bandwidth = random.uniform(min_bw, max_bw)
    else:
        bandwidth = random.uniform(0, min_bw)  # Low usage if inactive

    # Random spikes (5% chance)
    if random.random() < 0.05:
        bandwidth *= random.uniform(1.5, 2.5)  # Sudden spike
    return min(bandwidth, max_bw)  # Ensure max limit is not exceeded

def generate_latency():
    """Simulate network latency in milliseconds."""
    return round(random.uniform(10, 100), 2)  # Latency between 10-100ms

def generate_throughput():
    """Simulate data throughput in Mbps."""
    return round(random.uniform(5, 50), 2)  # Throughput between 5-50 Mbps

def generate_signal_strength():
    """Simulate signal strength in dBm (lower is worse)."""
    return round(random.uniform(-100, -50), 2)  # Signal strength between -100 to -50 dBm

def generate_data():
    """Continuously send simulated network data via socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"🚀 Data Generator Running on {HOST}:{PORT}, waiting for connection...")

        conn, addr = server.accept()
        print(f"✅ Connected to {addr}")

        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                total_bandwidth = 0  # Track total usage
                total_latency = 0
                total_throughput = 0
                total_signal_strength = 0

                for _ in range(NUM_USERS):
                    user_type = random.choice(list(USER_TYPES.keys()))
                    bandwidth = get_time_based_bandwidth(user_type)
                    latency = generate_latency()
                    throughput = generate_throughput()
                    signal_strength = generate_signal_strength()

                    total_bandwidth += bandwidth
                    total_latency += latency
                    total_throughput += throughput
                    total_signal_strength += signal_strength

                # Average values for the network
                avg_latency = round(total_latency / NUM_USERS, 2)
                avg_throughput = round(total_throughput / NUM_USERS, 2)
                avg_signal_strength = round(total_signal_strength / NUM_USERS, 2)

                # JSON response with three required features
                data = json.dumps({
                    "timestamp": timestamp,
                    "Latency (ms)": avg_latency,
                    "Data Throughput (Mbps)": avg_throughput,
                    "Signal Strength (dBm)": avg_signal_strength
                })

                conn.sendall(data.encode("utf-8"))
                print(f"📤 Sent Data: {data}")

                time.sleep(1)  # Send data every second
        except BrokenPipeError:
            print("⚠️ Connection closed. Restart the receiver to reconnect.")

if __name__ == "__main__":
    generate_data()
