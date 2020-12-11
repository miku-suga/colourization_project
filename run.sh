for i in {1..8}
do
	ls logs/saved/ -t |
		head -n 1 |
		xargs -I{} bash -c "python3 main.py 2>&1 -t1 -m logs/saved/{}/weights_model_1 -d logs/saved/{}/weights_discrim_1 | tee logs/stdout/postout{}.txt"
done
