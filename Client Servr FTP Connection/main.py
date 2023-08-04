# There are 10 types of people in this world. Those who understand binary and those who don't.
import ftplib
from os import listdir

if __name__ == '__main__':
    ftp_host = input("Enter server IP: ") or "localhost"
    ftp_port = int(input("Enter port number: ") or "21")
    ftp_user = input("Enter username: ") or "user"  # username = user
    ftp_pass = input("Enter password: ") or "pass"  # password = pass

    ftp = ftplib.FTP(timeout=30)
    ftp.connect(ftp_host, ftp_port)
    ftp.login(ftp_user, ftp_pass)

    while True:
        print("\n"
              "1. change directory.\n"
              "2. list of files in directory.\n"
              "3. download file.\n"
              "4. upload file.\n"
              "5. delete file.\n"
              "0. exit\n")
        choice = int(input("Enter your choice number: ") or "0")

        # Change Working Directory
        if choice == 1:
            path = ""
            while not path:
                path = input("Enter path: ")
            if path in ftp.nlst() or path == "/":
                status_code = ftp.cwd(path)
                if status_code.startswith("250"):
                    print(f"path changed to {ftp.pwd()} successfully.")
                else:
                    print("change directory failed.")
            else:
                print("path not found.")

        # Get List Of Files In Directory
        elif choice == 2:
            file_names = ftp.nlst()
            print(file_names)

        # Download File
        elif choice == 3:
            remote_file = input("Enter remote file name with extension to download: ") or ""
            if remote_file in ftp.nlst():
                with open(remote_file, "wb") as file:
                    status_code = ftp.retrbinary(f"RETR {remote_file}", file.write)
                if status_code.startswith("226"):
                    print("download successful.")
                else:
                    print("download failed.")
            else:
                print("file not found.")

        # Upload File
        elif choice == 4:
            local_file = input("Enter local file name with extension to upload: ") or ""
            if local_file in listdir():
                with open(local_file, "rb") as file:
                    status_code = ftp.storbinary(f"STOR {local_file}", file, blocksize=1024 * 1024 * 1024)
                if status_code.startswith("226"):
                    print("upload successful.")
                else:
                    print("upload failed.")
            else:
                print("file not found.")

        # Delete File
        elif choice == 5:
            remote_file = input("Enter remote file name with extension to delete: ") or ""
            if remote_file in ftp.nlst():
                status_code = ftp.delete(remote_file)
                if status_code.startswith("250"):
                    print("delete successful.")
                else:
                    print("delete failed.")
            else:
                print("file not found.")

        # Exit
        elif choice == 0:
            ftp.quit()
            print("FTP connection terminated")
            break

        else:
            print("wrong choice. try again.")
