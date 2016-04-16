include("data_loader.jl")
include("encoder.jl")
include("decoder.jl")

using Knet

img_cache = Dict()

function main()
    training_file = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/text/Flickr_8k.trainImages.txt"
    img_base_dir = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/Flicker8k_Dataset_Prep/"
    caption_file = "/mnt/kufs/scratch/cgumeli/NIC/Flickr8k/text/Flickr8k.token.txt"
    trn_imgs = image_names(training_file; make_set=true)
    captions = get_captions(trn_imgs, caption_file; base_dir=img_base_dir)
    shuffle!(captions)
    caption_txts = caption_texts(captions)
    caption_imgs = caption_image_names(captions)
    global dict = create_dict(caption_txts; min_freq=5)
    if true; return; end
    println("Dictionary size:", length(dict))
    global encoder = compile(:vgg16)
    global lstm = compile(:lstm_nic)
    global embedder = compile(:embedder)
    global generator = compile(:generator; dict_size=length(dict))
   
    train(encoder, embedder, lstm, generator, caption_imgs, caption_txts, dict)
    #@time train(net, caption_imgs, caption_txts, dict)
end

function train(encoder, embedder, lstm, generator, caption_imgs, caption_txts, dict;
               max_caption=39, gclip=0, batch_size=5, lr=0.1)

    reset!(embedder); reset!(lstm); reset!(generator)
    setp(encoder; lr = lr); setp(embedder; lr=lr); setp(lstm; lr=lr); setp(generator; lr=lr)
    
    for i = 1:batch_size:length(caption_txts)
        imgs, seqs = img_seq_pairs(caption_imgs[i:i+batch_size-1], caption_txts[i:i+batch_size-1], dict)
        #size(imgs) != (224, 224, 3, 5) && error("something bad!!", size(imgs))        
        encoding = forw(encoder, imgs)
        sforw(lstm, encoding) #the initial state        
        ystack = Any[] #output stack for the recurrent system
        for t=2:max_caption
            y_in = reshape(seqs[:,t-1,:], (size(seqs,1), size(seqs,3)))
            y_out = reshape(seqs[:,t,:], (size(seqs,1), size(seqs, 3)))
            push!(ystack, y_out)
            #recurrent forward steps
            embedding = sforw(embedder, y_in)
            lstm_output = sforw(lstm, embedding)
            ypred = sforw(generator, lstm_output)
        end
        
        while !isempty(ystack)
            ygold = pop!(ystack)
            #Backward steps of the recurrent layers
            gen_grad = sback(generator, ygold, softloss; getdx=true)
            lstm_grad = sback(lstm, gen_grad; getdx=true)
            sback(embedder, lstm_grad)
        end        
        enc_grad = sback(lstm; getdx=true) #the initial layer gradient
        update!(generator)
        update!(lstm; gclip=gclip)
        update!(embedder)
        #resets
        reset!(generator; keepstate=true)
        reset!(embedder; keepstate=true)
        reset!(lstm; keepstate=true)
        #cnn layer update
        back(encoder, enc_grad)
        update!(encoder)
    end
end

function generate_greedy(img, encoder, embedder, lstm, generator, dict; seq_limit=50)
    caption = repmat("", size(img,4))    
    dict_ary = dictasarray(dict)
    encoding = forw(encode, img)
    sforw(lstm, encoding)
    
    words = repmat(word2onehot("<sw>", dict), size(img,4))
    wemb = forw(embedder, words)
    lstm_out = forw(embedder, wemb)
    probs = convert(Array{Float, 2}, forw(generator, lstm_out))
    next_words = map((w)->dict_ary[w], findmax(probs)[2])

    done = false
    cnt = 0
    while !done && (cnt < seq_limit)
        done = true
        for i = 1:length(next_words)            
            next_words[i] != "<ew>" && (done = false)
            && (caption[i] = string(caption[i], " ", next_words[i]))            
        end
        cnt += 1
        wemb = forw(embedder, next_words)
        lstm_out = forw(embedder, wemb)
        probs = convert(Array{Float, 2}, forw(generator, lstm_out))
        next_words = map((w)->dict_ary[w], findmax(probs)[2])
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

    
