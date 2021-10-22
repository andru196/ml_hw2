RAW="data/raw"
LOG="download.log"
PROJ="home-credit-default-risk"
ZIP=$PROJ".zip"


rm -rf $RAW
date >> $LOG
kaggle competitions download -c $PROJ -p $RAW >> $LOG
unzip data/raw/$ZIP -d $RAW >> $LOG
rm data/raw/$ZIP >> $LOG
FILES="data/raw/*.zip"
for f in $FILES
do
	if test -f "$f"
	then
	
		unzip $f -d data/raw >> $LOG
		rm $f >> $LOG
		if [ $? -eq 0 ]
		then
			echo $f deleted
		fi
	fi
done
echo "HOMEWORK IN notebooks/HW2.ipynb"
