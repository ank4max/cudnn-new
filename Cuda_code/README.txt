To create Config File and data file
	python create_config_file_with_data.py --config cudnn --name=batchnormalization--batch_size 1 --channel 3 --height 8 --width 8
	
	
To create cudnn executable:
	nvcc cuda_batch_norm.cu  -o batchnormalization -lcudnn
	
	
To run program with data file
	python3 exe_batch_norm_cpp.py --testfile cudnn.test --data batchnorm_config2_data.txt 
	
	
To run program without data file (internaly generate random data)
	python3 exe_batch_norm_cpp.py --testfile cudnn.test
	
