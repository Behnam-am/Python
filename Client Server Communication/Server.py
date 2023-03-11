import socket
from random import randint

if __name__ == '__main__':
    file_size = int(1024e6)
    extension = {"music": "mp3", "picture": "jpg", "outstanding statement": "txt"}
    first_choice = ""
    second_choice = ""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket successfully created.")
    port = 54321
    s.bind(('localhost', port))
    print(f"socket bound to {port}.")
    s.listen(5)
    print("socket is listening...")
    server, address = s.accept()
    print('Got connection from', address)

    while True:
        message = server.recv(file_size).decode()
        print(f"Client says: {message}")
        if message == "end":
            print(f"terminating connection from {address}")
            break
        elif message in ["Hello", "hello"]:
            server.send('Hello'.encode())
        elif message == "What do you have for me today?":
            server.send('Choose one of these topics: kindness, knowledge, morality'.encode())
        elif message in ["kindness", "knowledge", "morality"]:
            first_choice = message
            server.send(
                'Choose one of these options about the selected topic: music, picture, outstanding statement'.encode())
        elif message in ["music", "picture", "outstanding statement"] and first_choice:
            second_choice = message
            file = open(f"server/{first_choice}/{second_choice}/{randint(1, 3)}.{extension[second_choice]}", "rb")
            line = file.read(file_size)
            server.send(line)
            file.close()
            print('File has been sent successfully.')
        else:
            server.send('Unknown command.'.encode())

    server.close()
