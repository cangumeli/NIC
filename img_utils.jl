using Images, ImageMagick

#takes an image as input and transform it into an array
function img2vec(img; crop=true, crop_target=224, rescale = false, minmin_scale=256, minmax_scale=256, flip=true)
    rescale && (img = random_rescale(img; srange=minmin_scale:minmax_scale))
    if crop
        x_start = rand(1:size(img,1)-crop_target+1)
        y_start = rand(1:size(img,2)-crop_target+1)
        img = subim(img, "x", x_start:x_start+crop_target-1, "y", y_start:y_start+crop_target-1)
    end
    flip && rand(1:20)==1 && flipdim(img, 1)
    r = convert(Array{Float16, 3}, raw(img'))
    I = zeros(Float16, size(r)[2], size(r)[3], 3, 1)
    I[:,:,1,1] = r[1,:,:]
    I[:,:,2,1] = r[2,:,:]
    I[:,:,3,1] = r[3,:,:]
    return I
end

#Apply mean subtraction to the image
function mean_subtract(img)
    return img .- mean(img,3)
end

#Random rescaling system that applies isomorphic rescaling
function random_rescale(img; srange=256:512)
    S = rand(srange)
    s1, s2 = size(img)
    scale = S/min(s1,s2)
    s1 = convert(Int, ceil(s1*scale))
    s2 = convert(Int, ceil(s2*scale))

    return Images.imresize(img, (s1, s2))
end


#Random cropping operation
function random_crop(img_vector; target=(224,224))
    cropped = zeros(target[1], target[2], size(img_vector,3), size(img_vector,4))
    for i = 1:size(img_vector,3)        
        d1_start = rand(1:size(img_vector,1)-target[1]+1)
        d2_start = rand(1:size(img_vector,2)-target[2]+1)    
        cropped[:, :, i, :] = img_vector[d1_start:d1_start+target[1]-1,
                                         d2_start:d2_start+target[2]-1, i, :]
    end
    return cropped
end


#some dummy Knet models for testing image processing functions
#using Knet
#@knet function conv_test(x)
#    w = par(init=Xavier(), dims=(3,3,0,3))
#    return conv(w, x)
#end


#@knet function convtest(x)
#    w = par(init=rand(3,3,0,3))
#    return conv(w, x; padding=1)
#end
