import socket
import time

if __name__ == '__main__':
    file_size = int(1024e6)
    extension = {"music": "mp3", "picture": "jpg", "outstanding statement": "txt"}
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 54321
    client.connect(('localhost', port))

    while True:
        text = (input('Enter your request: ') or " ")
        client.send(text.encode())

        if text == "end":
            print("terminating connection...")
            break
        elif text in ["music", "picture", "outstanding statement"]:
            file = open(f"client/{int(time.time())}.{extension[text]}", "wb")
            line = client.recv(file_size)
            file.write(line)
            file.close()
            print(f"{text} file received.")
        else:
            message = client.recv(file_size).decode()
            print(f"Server says: {message}")

    client.close()
