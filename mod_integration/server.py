import socket
import re
import luadata

# Set up server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 12345))  # Localhost, port 12345
server.listen(1)
print("Waiting for connection...")
conn, addr = server.accept()
print("Connected by", addr)

# Main loop
while True:
    data = conn.recv(1024).decode().strip()  # Receive game state
    if data:
        # print("Raw data:", data)
        # Preprocess to fix numeric keys: convert "number =" to "[number] ="
        fixed_data = re.sub(r'(\s*)(\d+)\s*=', r'\1[\2] =', data)
        # print("Fixed data:", fixed_data)
        try:
            game_state = luadata.unserialize(fixed_data)
        except Exception as e:
            print("Error unserializing data:", e)
            continue

        print("Game state:", game_state)

        try:
            score = int(game_state["score"])
        except ValueError:
            score = 0

        action = "play_card" if score > 50 else "skip"
        conn.send((action + "\n").encode())
