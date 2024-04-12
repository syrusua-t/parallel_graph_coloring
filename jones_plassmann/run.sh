input="../inputs/amazon0601.txt"
output="../outputs/output.txt"
mode="minmax"

make > ./tmp
./cudajp -f $input -o $output -v -m $mode
make clean > ./tmp
rm ./tmp
echo "==========result=========="
python3 ../validate/validate.py -i $input -c $output
echo "=========================="
echo ""
