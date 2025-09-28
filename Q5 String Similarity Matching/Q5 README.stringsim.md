###### ***Steps to Run the Code***



1\. Install Python



Make sure you have Python 3.8 or later installed.



You can check with:

python --version



2\. Download or copy the script



Save the code as string\_similarity.py inside a folder (e.g., C:\\Users\\Aosaf\\Python\_Files\\).



3\. Open PowerShell or Command Prompt



Navigate to the folder where you saved the script:

cd "C:\\Users\\Aosaf\\Python\_Files"



4\. Run the script

python string\_similarity.py



5\. Input your strings



Enter two strings between 6–10 characters each when prompted.



The program will:



Calculate similarity percentage (Levenshtein + alignment).



Show aligned strings.



Display which characters match, mismatch, or require gaps.



###### ***Dependencies Required***



Python Standard Library only



difflib (comes with Python)



No external packages are required.



###### ***Assumptions Made***



Input strings must be between 6–10 characters.



Comparison is case-insensitive by default (so "Hello" and "hello" are treated as the same).



Alignment uses a simple scoring system:



&nbsp; Match = +1



&nbsp; Mismatch = 0



&nbsp; Gap = –1



The similarity percentage is calculated in two ways:



&nbsp; Levenshtein similarity → based on edit distance.



&nbsp; Alignment similarity → based on number of character matches after alignment.

