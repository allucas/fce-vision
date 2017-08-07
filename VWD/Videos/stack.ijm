args = split(getArgument(),","); 
count = args[1];
dir = args[0]
for (i=1000 ; i<count; i++){
open(dir+"/"+i+".avi");
run("Image Sequence... ", "format=TIFF name=flow start=1000 save="+dir+"/"+i);
run("Close");
}

