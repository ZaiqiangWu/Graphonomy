# Example of inference
#python exp/inference/inference.py  --loadmodel ./universal_trained.pth --img_path ./ichao_input.jpg --output_path ./img/ --output_name /output_file_name.jpg
#python exp/inference/inference.py  --loadmodel ./cihp2pascal.pth --img_path ./ichao_input.jpg --output_path ./img/ --output_name /output_file_name.jpg
python exp/inference/inference.py  --loadmodel ./inference.pth --img_path ./ichao_input.jpg --output_path ./img/ --output_name /output_file_name.jpg