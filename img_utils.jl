using Images, ImageMagick

#takes an image as input and transform it into an array
function img2vec(img)
    r = convert(Array{Float64, 3}, raw(img))
    I = zeros(size(r)[2], size(r)[3], 3, 1)
    I[:,:,1,1] = r[1,:,:]
    I[:,:,2,1] = r[2,:,:]
    I[:,:,3,1] = r[3,:,:]
    return I
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
