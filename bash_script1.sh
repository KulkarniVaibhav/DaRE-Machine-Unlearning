

#!/bin/bash
echo "New batch \n \n" >> results.txt
python3 test.py bank_marketing 32000 8000 10
python3 test.py bank_marketing 32000 8000 5000
python3 test.py bank_marketing 32000 8000 16000
python3 test.py adult 32000 8000 10
python3 test.py adult 32000 8000 5000
python3 test.py adult 32000 8000 16000
python3 test.py AppBehaviour 40000 10000 10
python3 test.py AppBehaviour 40000 10000 6000
python3 test.py AppBehaviour 40000 10000 20000



