python3 main_rodie2.py

while read line;
do
  for x in $line;
  do
    python3 evaluate3.py --split_select $x
  done
done < /home/gauthierv/jodie/hyper-parameter.txt