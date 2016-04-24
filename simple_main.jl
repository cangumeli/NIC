include("data_loader.jl")
include("encoder.jl")
include("decoder.jl")

using Knet
using JLD

function main(;encoding_required=false, load_file="vgg_encodings1.jld", epochs=100)
    training_file = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/text/Flickr_8k.trainImages.txt"
    img_base_dir = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/Flicker8k_Dataset_Prep224/"
    caption_file = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/text/Flickr8k.token.txt"

    trn_images = image_names(training_file; make_set=true)
    captions = get_captions(trn_images, caption_file; base_dir=img_base_dir, cap_cnt=1)
    cap_txts = caption_texts(captions)
    cap_imgs = caption_image_names(captions)
    
    global dict = create_dict(cap_txts)                

    cnn = nothing
    if encoding_required
        cnn = compile(:vgg16)
        encode_images(cnn, cap_imgs, dict)
    end
    #return
    
    global encoder = compile(:top_layer)
    global embedder = compile(:embedder)
    global lstm = compile(:lstm_nic; dict_size=length(dict))

    #return
    max_cap = largest_caption(cap_txts, dict)
    
    img_encs = JLD.load(load_file, "encodings")
    for epoch = 1:epochs
        println("epoch: ", epoch)
        @time train(img_encs, cap_txts)
       # train("encodes.jld; max_cap=max_cap)
        println()
    end
    
end

function encode_images(cnn, img_names, dict; target_file="vgg_encodings1.jld", batch_size=20)
    encs =zeros(Float32, 4096, length(img_names))

    for i=1:batch_size:length(img_names)
        imgs = get_images(img_names[i:i+batch_size-1], Float32(114.79933333333334))
        encs[:, i:i+batch_size-1] = encode(cnn, imgs)
        println(i)
    end
    JLD.save(target_file, "cnn", clean(cnn),
             "encodings", encs, "filenames", img_names)
end


function train(img_encs, caption_txts; max_cap=39, batch_size=100, gclip=0, lr=0.1)
    reset!(lstm)
    reset!(embedder)

    setp(lstm; lr=lr)
    setp(embedder; lr=lr)
    setp(encoder; lr=lr)
    
    for i = 1:batch_size:length(caption_txts)
        seqs, masks = get_sequences(caption_txts[i:i+batch_size-1], dict; max_caption=max_cap)
        
        encoding = forw(encoder, img_encs[:, i:i+batch_size-1])
        sforw(lstm, encoding; first=true)
        ystack = Any[] #output stack for the recurrent system
        for t=2:max_cap
            y_in = reshape(seqs[:,t-1,:], (size(seqs,1), size(seqs,3)))
            y_out = reshape(seqs[:,t,:], (size(seqs,1), size(seqs, 3)))                        
            push!(ystack, y_out)
            emb = sforw(embedder, y_in)
            sforw(lstm, emb)
        end

        while !isempty(ystack)
            mask = masks[:, length(ystack)]
            ygold = pop!(ystack)
            #Backward steps of the recurrent layers
            lstm_grad = sback(lstm, ygold, softloss; mask=mask, getdx=true)
            sback(embedder, lstm_grad; mask=mask)
        end

        enc_grad = sback(lstm; getdx=true) #the initial layer gradient
        back(encoder, enc_grad)
        
        update!(lstm; gclip=gclip)
        update!(embedder)
        update!(encoder)
        
        reset!(lstm)
        reset!(embedder)
    end
end

function generate_greedy(cnn, img, dict_ary; seq_limit=50)
    reset!(embedder)    
    reset!(lstm)

    caption = repmat([""], size(img, 4))

    img_rep = forw(cnn, img)
    encoding = forw(encoder, img_rep)
    forw(lstm, encoding; first=true)

    words = word2onehot(repmat(["<sw>"], size(img,4)), dict)
    
    wembs = forw(embedder, words)
    probs = convert(Array{Float32,2}, forw(lstm, wembs))
    
    cnt = 0        
    done = false
    next_words = map((w)->dict_ary[w], findmax(probs,1)[2])
    
    while !done && (cnt < seq_limit)
        done = true
        for i = 1:length(next_words)
            if next_words[i] != "<ew>"
                done = false
                caption[i] = string(caption[i], " ", next_words[i])
            end
        end
        cnt += 1
        words = word2onehot(next_words, dict)
        embs = forw(embedder, words)
        probs = convert(Array{Float32, 2}, forw(lstm,embs))
        next_words = map((w)->dict_ary[w], findmax(probs, 1)[2])
    end
    
    return caption
end

