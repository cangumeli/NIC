#"Module contains the encoder CNN model"
#module Encoder

using Knet

@knet function vgg_generic_conv(x; cinput=0, coutput=0)
    w = par(init=Xavier(), dims=(3, 3, cinput, coutput))
    b = par(init=Constant(0), dims=(1,1,coutput,1))
    c = conv(w, x; padding=1)
    return relu(c)
end

@knet function layer_2conv(x0; output=64)
    x1 = vgg_generic_conv(x0; coutput=output)
    x2 = vgg_generic_conv(x1; coutput=output)
    return pool(x2)
end

@knet function layer_3conv(x0; output=64)
    x1 = vgg_generic_conv(x0; coutput=output)
    x2 = vgg_generic_conv(x1; coutput=output)
    x3 = vgg_generic_conv(x2; coutput=output)
    return pool(x3)
end

@knet function layer_fcdrop(x0)
    x1 = wbf(x0; f=:relu, out=4096)
    return drop(x1; pdrop=0.5)
end


@knet function vgg16(xl0; output=512)
    #Convolutional layers
    xl1 = layer_2conv(xl0; output=64)    
    xl2 = layer_2conv(xl1; output=128)
    xl3 = layer_3conv(xl2; output=256)    
    xl4 = layer_3conv(xl3; output=512)    
    xl5 = layer_3conv(xl4; output=512)

    #Fully Connected layers
    xl6 = layer_fcdrop(xl5)
    xl7 = layer_fcdrop(xl6)

    #Embedding layer for image representation
    return wdot(xl7; out=output)

    # return wbf(xl7; f=:soft, out=output)
end
