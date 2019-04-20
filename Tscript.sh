counter=1
while inotifywait -q -e create testModel/ >/dev/null; do
    echo $counter
    counter=$((counter+1))
    cp -rf models/ models$counter 
        # do whatever else you need to do
done
