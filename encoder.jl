#"Module contains the encoder CNN model"
#module Encoder

using Knet

include("weigth_initializer.jl")
ws, bs = get_weights()

#bs[1] = reshape(bs[1], (1,1,64,1))
#ws = reshape(ws, (length(ws),1))

#bs = reshape(bs, (length(bs),1))

@knet function vgg_generic_conv(x; winits=ws, binits=bs, wi=1, bi=1, out=64)
    w = par(init=winits[wi], dims=size(winits[wi]))                                              
    b = par(init=binits[bi], dims=size(binits[bi]))
    
    c = conv(w, x; padding=1)
    cb = c + b
    return relu(cb)
end

@knet function layer_2conv(x0; start_index=1)
    x1 = vgg_generic_conv(x0; wi=start_index, bi=start_index)
    x2 = vgg_generic_conv(x1;  wi=start_index+1, bi=start_index+1)
    return pool(x2)
end

@knet function layer_3conv(x0; start_index=1)
    x1 = vgg_generic_conv(x0; wi=start_index, bi=start_index)
    x2 = vgg_generic_conv(x1; wi=start_index+1, bi=start_index+1)
    x3 = vgg_generic_conv(x2; wi=start_index+2, bi=start_index+2)
    return pool(x3)
end

@knet function layer_fcdrop(x0; winits=ws, binits=bs, start_index=1)
    x1 = wbf(x0; f=:relu,
             winit=winits[start_index],
             binit=binits[start_index]
             )
#    w = par(init=winits[start_index], dims=size(winits[start_index]))
    #x1 = w * x0
    return drop(x1; pdrop=0.5)
end

#@knet function layer_fconv(x0; winits=ws, binits=bs, start_index=1)
    #x1 = cbfp(x0; cinit=ws[start_index], binit=bs[start_index], f=:relu)
#    x1 = wconv(x0; cinit=winits[start_index])
#    x2 = bias(x1; binit=binits[start_index])
#    return drop(x1; pdrop=0.5)
#end

@knet function layer_fconv(x; winit=ws,binit=bs, start_index=14)
    w = par(init=winit[start_index], dims=size(ws[start_index]))
    b = par(init=binit[start_index], dims=size(bs[start_index]))
    
    x1 = conv(w, x)
    x2 = x1 + b
    return x2
end

              
@knet function vgg16(xl0)
    #Convolutional layers
    xl1 = layer_2conv(xl0;  start_index=1) 
    xl2 = layer_2conv(xl1;  start_index=3)
    xl3 = layer_3conv(xl2;  start_index=5)    
    xl4 = layer_3conv(xl3;  start_index=8)
    xl5 = layer_3conv(xl4;  start_index=11)

    #return xl5
    #Fully Connected layers
    if train
        xl6 = layer_fcdrop(xl5; start_index=14)
        xl7 = layer_fcdrop(xl6, start_index=15)              
    else
        xl6 = layer_fconv(xl5; start_index=14)
        xl7 = layer_fconv(xl6; start_index=15)
    end    
    return xl7
    #return wdot(xl7; out=output)

    # return wbf(xl7; f=:soft, out=output)
end

@knet function top_layer(x;output=512)
    return wdot(x; out=output)
end

f = compile(:vgg16)
x = forw(f, rand(Float32,256, 320, 3, 5))

#ft = compile(:test)
#forw(ft, x)
#f2 = compile(:top_layer)
#forw(f2, x)

#@knet function test(x;out=100)    
 #   return wdot(x; out=100)
#end


