rm -rf ./tmp
mkdir -p ./tmp/
for file_name in ../prores_decoder/test/input_sample10bit/*.yuv ;do
    OUT_NAME=`basename $file_name`
    #../../prores_encoder/encoder -l ../../prores_encoder/luma_matrix.txt -c ../../prores_encoder/chroma_matrix.txt -q ../../prores_encoder/qscale_128x16.txt -h 128 -v 16 -m 8 -i ${file_name}  -o ./tmp/${OUT_NAME%.yuv}.bin  
    ./sample_app/encoder -l ./luma_matrix.txt -c ./chroma_matrix.txt -q ./qscale_128x16.txt -h 128 -v 16 -m 8 -i ${file_name}  -o ./tmp/${OUT_NAME%.yuv}.bin  
#    ../../prores_decoder/decoder ./tmp/${OUT_NAME%.yuv}.bin ./tmp/${OUT_NAME%.yuv}_dec.yuv 
#   diff ./tmp/${OUT_NAME%.yuv}_dec.yuv ./input_128x16/${OUT_NAME%.yuv}_dec.yuv
    diff ./tmp/${OUT_NAME%.yuv}.bin ../prores_decoder/test/output_sample10bit/${OUT_NAME%.yuv}.bin
    #../sn16/sn16 ${file_name} ./tmp/${OUT_NAME%.yuv}_dec.yuv 128 16
#	break
done
#./all_test/tt0.sh &
#./all_test/tt1.sh &
#./all_test/tt2.sh &
#./all_test/tt3.sh &
#./all_test/tt4.sh &
#./all_test/tt5.sh &
#./all_test/tt6.sh &
#./all_test/tt7.sh &
#./all_test/tt8.sh &
#./all_test/tt9.sh &
