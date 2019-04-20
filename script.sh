counter=1
while inotifywait -q -e create models/ >/dev/null; do
    
   	sleep 10 
	echo $counter
    	cp -rf models/ m/models$counter
        
    	counter=$((counter+1))
	# do whatever else you need to do
done
