Copy the contents of a file to clipboard:

cat file.txt | xclip -selection c

Copy the contents of the clipboard to a file:

xclip --selection c -o > file.txt
