"Module that contains decoder lstm model"
module Decoder
using Knet

@knet function embedder(x; output=512)
    return wdot(x; out=512)
end

@knet function decoder(x; dict_size=0)    
    h = lstm(x; out=dict_size)
    return soft(h)
end
