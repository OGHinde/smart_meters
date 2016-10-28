

for i in $(seq 2 20); 
	do for j in $(seq 2 10); 
		do python main_validation.py weekly $i $j; 
	done; 
done;