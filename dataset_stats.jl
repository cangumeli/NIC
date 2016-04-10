#Module DatasetStats

using Images, ImageMagick

f = open("Flickr8k/text/Flickr_8k.trainImages.txt")

cnt = 0
maxr, maxc = 0, 0
minr, minc = Inf, Inf

for l in eachline(f)
    print(l)
    cnt = cnt + 1
    img = load(string("/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/images/", l[1:length(l)-1]))
    (r,c) = size(img)
    
    maxr = max(maxr, r)
    minr = min(minr, r)
    maxc = max(maxc, c)
    minc = min(minc, c)
end

close(f)

println("Max row: ",maxr)
println("Max col: ", maxc)
println("Min row: ", minr)
println("Min col: ", minc)

