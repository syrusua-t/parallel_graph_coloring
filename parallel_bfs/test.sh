make
./ompbfs -f ../inputs/amazon0601.txt -m s -o ../outputs/output.txt
make clean
python3 ../validate/validate.py