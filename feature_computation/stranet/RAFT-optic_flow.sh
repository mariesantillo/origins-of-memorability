#use RAFT Opticflow (https://github.com/spmallick/learnopencv/tree/master/Optical-Flow-Estimation-using-Deep-Learning-RAFT)
#run
python3 inference.py --model ./models/raft-sintel.pth --video_dir /home/mariesantillo/RAFT/foundcog_videos --save 
#This customized script produces the frame, the optic .flo files and the videos of the optic flow
