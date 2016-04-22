using MAT, Knet

function get_weights(dst="/mnt/kufs/scratch/cgumeli/NIC/imagenet-vgg-verydeep-16.mat", fc_start=Inf, weight_type=Float32)
    file = matopen(dst)
    global layers = read(file, "layers")
    ws = Any[]
    bs = Any[]
    for i=1:length(layers)
        l = layers[i]
        if haskey(l, "weights")
            #w = l["weights"][1]
            #b = l["weights"][2]
            if length(layers) >= fc_start && false
                sw = size(l["weights"][1])
                l["weights"][1] = reshape(l["weights"][1],
                                          (sw[1]*sw[2]*sw[3],sw[4]))'
                #l["weights"][2] = reshape(l["weights"][1],                                      
                #println(size(w))
                #b = reshape(b, (length(b),1))
                #println(size(b))
                push!(ws, convert(Array{weight_type,2},l["weights"][1]))
            end

            if length(ws) < fc_start-1
               # println(i)
                push!(ws, convert(Array{weight_type,4},l["weights"][1]))
            #push!(ws, convert(Array{weight_type,4},l["weights"][2]))                            
                push!(bs, convert(Array{weight_type,4},reshape(l["weights"][2], (1,1,length(l["weights"][2]),1))))
            else #fully connected
                sw = size(l["weights"][1])
                sw_ = (sw[1]*sw[2]*sw[3], sw[4])
                push!(ws,
                      convert(Array{weight_type, 2},
                              reshape(l["weights"][1], sw_)')) #make w 2-D
                push!(bs, convert(Array{weight_type,1},
                                  l["weights"][2][:])) #make b 1-D                
                #println(size(last(ws)))
                #println(size(last(bs)))
                #println()
            end
            
        end
    end 
    close(file)
    return ws, bs
end
    
