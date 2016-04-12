using Knet
include("encoder.jl")
include("decoder.jl")

@knet function nic(x; encoder=:vgg16, embedding_space=512, dict=0)
    if image_in
        xt = encoder(x; output=embedding_space)
    else
        xt = embedder(x; output=embedding_space)
    end

    return decoder(xt; mem_size=embedding_space, dict_size=dict)
end
