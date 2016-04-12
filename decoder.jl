#"Module that contains decoder lstm model"

using Knet

@knet function embedder(x; output=512)
    return wdot(x; out=512)
end

@knet function decoder(x; mem_size=512, dict_size=0)    
    h = lstm(x; out=mem_size)
    return wbf(h; f=:soft, out=dict_size)
end
