python3 main_ray.py

while read line;
do
  for x in $line;
  do
    python3 evaluate_model.py --split_select $x
  done
done < /home/gauthierv/jodie/hyper-parameter.txt