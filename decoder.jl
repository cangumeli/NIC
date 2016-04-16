#"Module that contains decoder lstm model"

using Knet

@knet function embedder(x; output=512)
    emb = wdot(x; out=output)
    return emb
end

@knet function wf2(x, y; f=:sigm, output=512)
    term1 = wdot(x; out=output)
    term2 = wdot(y; out=output)
    sm = term1 + term2
    gout = f(sm)
    return gout
end


@knet function lstm_nic(x; mem_size=512, dict_size=0)    
    i = wf2(x, m; output=mem_size)
    f = wf2(x, m; output=mem_size)
    o = wf2(x, m; output=mem_size)
    h = wf2(x, m; f=:tanh, output=mem_size)
    c = f .* c + i .* h
    m = o .* c
    if seq
        p = generator(m; dict_size=dict_size)
        return p
    else
        return m
    end
end


@knet function generator(x;  dict_size=0)
    probs =  wbf(x; f=:soft, out=dict_size)
    return probs
end


#@knet function decoder(x;  mem_size=512, emb_space=512, dict=0)
#    #Equation 10-11
#    if out_layer
#        xt = (x; output=emb_space)
#    else
#        xt = x
#    end

    #Equation 12
#    lout = lstm_nic(xt; output=mem_size)

 #   if out_layer
 #       gn = generator(lot; dict_size=dict)
 #       return gn
 #   else
 #       return lout
 #   end
    
#end
