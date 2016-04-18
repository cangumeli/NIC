include("data_loader.jl")
include("encoder.jl")
include("decoder.jl")

using Knet
using JLD

img_cache = Dict()
debug = false

function main()
    training_file = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/text/Flickr_8k.trainImages.txt"
    img_base_dir = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/Flicker8k_Dataset_Prep/"
    caption_file = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/text/Flickr8k.token.txt"
    trn_imgs = image_names(training_file; make_set=true)
    captions = get_captions(trn_imgs, caption_file; base_dir=img_base_dir, cap_cnt=1) #get only one caption per image in training

    if debug
        shuffle!(captions)
        captions = randsubseq(captions, 0.5)[1:1000]
    end
    
    #println(size(captions))
    caption_txts = caption_texts(captions)
    caption_imgs = caption_image_names(captions)
    global dict = create_dict(caption_txts; min_freq=5)
    #if true; return; end
    println("Dictionary size:", length(dict))
    global encoder = compile(:vgg16)
    global embedder = compile(:embedder)
    global lstm = compile(:lstm_nic; dict_size=length(dict))
    #global embedder = compile(:embedder)
    #global generator = compile(:generator; dict_size=length(dict))
    epochs = 5
    println("images: ", length(caption_imgs))
    println("captions: ", length(caption_imgs))
    for epoch = 1:epochs
        println("epoch:", epoch)                
        train(encoder, lstm, caption_imgs, caption_txts, dict)
    end    
    #@time train(net, caption_imgs, caption_txts, dict)
end

function backup()
    #JLD.save("models_e1.jld", "encoder", clean(encoder), "lstm", clean(lstm),
    #        "embedder", clean(embedder), "generator", clean(generator))
    JLD.save("experiment1.jld", "embedder", embedder, "encoder", encoder, "lstm", lstm, "dict", dict)
end

function train(encoder, lstm, caption_imgs, caption_txts, dict;
               max_caption=39, gclip=0, batch_size=20, lr=0.1, backup_freq=100)
    
    reset!(lstm)
    reset!(embedder)
    setp(embedder; lr=lr)
    setp(encoder; lr=lr)
    setp(lstm; lr=lr)            
    bc = 0
    for i = 1:batch_size:length(caption_txts)
        imgs, seqs, masks = img_seq_pairs(caption_imgs[i:i+batch_size-1], caption_txts[i:i+batch_size-1], dict)
        #size(imgs) != (224, 224, 3, 5) && error("something bad!!", size(imgs))
        bc += 1
        if rem(bc,100) == 0
            println("backup @", bc)
            backup()
        end
        
        encoding = forw(encoder, imgs)
        sforw(lstm, encoding; first=true) #the initial state 
        ystack = Any[] #output stack for the recurrent system        
        for t=2:max_caption
            y_in = reshape(seqs[:,t-1,:], (size(seqs,1), size(seqs,3)))
            y_out = reshape(seqs[:,t,:], (size(seqs,1), size(seqs, 3)))
#            println(size(y_in))
 #           println(size(y_out))
            push!(ystack, y_out)
            #recurrent forward step
            emb = sforw(embedder, y_in)
            sforw(lstm, emb)
#            embedding = sforw(embedder, y_in)
 #           lstm_output = sforw(lstm, embedding)
  #          ypred = sforw(generator, lstm_output)
        end
        
        while !isempty(ystack)
            mask = masks[:, length(ystack)]
            ygold = pop!(ystack)
            #Backward steps of the recurrent layers            
            #gen_grad = sback(generator, ygold, softloss; getdx=true, mask=mask)
            #lstm_grad = sback(lstm, gen_grad; getdx=true, mask=mask)
            #sback(embedder, lstm_grad; mask=mask)            
            lstm_grad = sback(lstm, ygold, softloss; mask = mask, getdx=true)
            sback(embedder, lstm_grad)
        end
        
        enc_grad = sback(lstm; getdx=true) #the initial layer gradient
        #update!(generator)
        update!(lstm; gclip=gclip)
        #update!(embedder)
        #resets
        #reset!(generator)
        #reset!(embedder)
        reset!(lstm)
        #cnn layer update
        back(encoder, enc_grad)
        update!(encoder)
    end
end

function generate_greedy(img, encoder, embedder, lstm, dict; seq_limit=50)
    reset!(lstm)
    reset!(embedder)
    caption = repmat("", size(img,4))    
    dict_ary = dictasarray(dict)
    encoding = forw(encoder, img)
    forw(lstm, encoding; first=true)
    
    words = word2onehot(repmat(["<sw>"], size(img,4)), dict)
    #wemb = forw(embedder, words)
    wembs = forw(embedder, words)
    probs = convert(Array{Float32,2}, forw(lstm,wembs))
    #probs = convert(Array{Float, 2}, forw(generator, lstm_out))
    next_words = map((w)->dict_ary[w], findmax(probs,1)[2])
    done = false
    cnt = 0
    while !done && (cnt < seq_limit)
        done = true
        for i = 1:length(next_words)            
            (next_words[i] != "<ew>") & (done = false) & (caption[i] = string(caption[i], " ", next_words[i]))
        end
        cnt += 1
        words = word2onehot(next_words)
        embs = forw(embedding, words)
        probs = convert(Array{Float32, 2}, forw(lstm, words))
        #wemb = forw(embedder, next_words)
        #lstm_out = forw(embedder, wemb)
        #probs = convert(Array{Float, 2}, forw(generator, lstm_out))
        next_words = map((w)->dict_ary[w], findmax(probs, 1)[2])
    end

    
    #for i = 1:length(next_words)
    #    next_word = next_words[i]
    #    while next_word != "<ew>"
    #        caption[i] = string(caption[i], " ", next_word)
     #       wemb = forw(embedder, word2onehot(next_word, dict))
      #      lstm_out = forw(embedder, wemb)
       #     probs = forw(generator, lstm_out)                
        #    next_word = dict_ary[findmax(probs)[2]]
       # end
    #end
    return caption
end



function beam_search(img, encoder, embedder, lstm, generator, dict; k=20)    
    function nextp(next_word)
        wemb = forw(embedder, word2onehot(next_word, dict))
        lstm_out = forw(lstm, wemb)
        probs = forw(generator, lstm_out)
    end
    dict_ary = dictasarray(dict)
    enc = forw(encoder, img)
    #sforw(
            
    #TODO: implement beam search
    
end

#returns the indices of largest k in a vector
function largest_k(vct, k)
    max_inds = zeros(Int, k, 1)
    temp_vct = zeros(length(vct), 1)
    copy!(temp_vct, vct)
    for i=1:k
        max_inds[i] = findmax(temp_vct)[2]
        temp_vct[max_inds[i]] = -Inf
    end
    return max_inds
end
        
function bleu1(spred, sgold)
    
end

function train_old(encoder, embedder, decoder, generator, cap_imgs, cap_txt, dict; lr=0.01, cache=true, batch_size=100, max_caption=39, nforw=100, gclip=0)
    
    for i = 1:batch_size:length(cap_txt)        
        imgs, seqs = img_seq_pairs(cap_imgs[i:i+batch_size-1], cap_txt[i:i+batch_size-1], dict; img_cache=nothing)
        #println("iter:",iter)
        #println(size(imgs))
        #println(size(seqs))
        #iter += 1        
        println(size(imgs))
        x = imgs
        ystack = Any[]        
        for t = 1:max_caption
            println(t)
            y = reshape(seqs[:,t,:], (size(seqs,1), size(seqs,3)))
            forw(net, x; image_in=(t==1))
            push!(ystack, y)            
            x = y
        end
        while !isempty(ystack)
            ygold = pop!(ystack)
            back(net, ygold, softloss)
        end
        update!(net; gclip=gclip)
        reset!(net; keepstate=true)
    end
end
main()
