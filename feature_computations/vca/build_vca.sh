#convert videos to y4m format 
ffmpeg -i sample.mp4 -c:v rawvideo -pix_fmt yuv420p out.y4m

#build VCA based on Manon et al repo (https://github.com/cd-athena/VCA/blob/stable/docs/index.md)
#Run via VCA.exe with block size 32x32

.\vca.exe --input C:\Users\MSANTILL\Desktop\VCA\videos\minions_supermarket.y4m --complexity-csv C:\Users\MSANTILL\Desktop\VCA\results\test\minions_supermarket_y4m32.csv --block-size 32
