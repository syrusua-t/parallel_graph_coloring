input="../inputs/amazon0601.txt"
output="../outputs/output.txt"
mode="multihash"
strategy="double"

make > ./tmp
./cudajp -f $input -o $output -v -m $mode -s $strategy
make clean > ./tmp
rm ./tmp
echo "==========result=========="
python3 ../validate/validate.py -i $input -c $output
echo "=========================="
echo ""
