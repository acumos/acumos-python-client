#! /bin/bash

header=".. THIS FILE WAS GENERATED. DO NOT EDIT.\n"

for mdfile in $(find . -name '*.md'); do
	outfile="${mdfile%.*}.rst"
	pandoc --from=markdown --to=rst --output=$outfile $mdfile;
	echo -e $header | cat - $outfile > temp && mv temp $outfile
done
