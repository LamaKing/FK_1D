#!/bin/bash
src_fld=$HOME/Documents/Post_doc-SISSA/heat_ring/MD_code/
src_fname=(  FK_1D.py create_chain.py driver.py RK45_lang.py )

force_opt=""
if [[ $1 == "f" ]]
then
    echo "Force option enabled"
    force_opt="-f"
fi

echo "link from $src_fld to $PWD "; 
for i in ${src_fname[@]}
do 
        echo "linking $i"
        ln $force_opt -s $src_fld/$i ; 
done
