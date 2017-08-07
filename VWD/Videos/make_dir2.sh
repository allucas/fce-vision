a=1000
for i in *.avi; do
  new=$(printf "%d.avi" "$a") 
  #mv -i -- "$i" "$new" #use this to avoid overwriting
  mv -- "$i" "$new"
  mkdir -p $a
  ffmpeg -i $a".avi" $a/img1%04d.jpg -hide_banner
  let a=a+1
done
