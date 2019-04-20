counter=1
while inotifywait -q -e create models/ >/dev/null; do
    
   	sleep 10 
	echo $counter
    	counter=$((counter+1))
    	cp -rf models/ m/models$counter
        # do whatever else you need to do
done
