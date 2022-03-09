#!/bin/bash
for i in {100..3000..72}
do 
	python3 test.py AppBehaviour 20000 10 $i 
done

