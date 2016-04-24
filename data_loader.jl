include("img_utils.jl")

function image_names(filename; erase_newline=true, make_set=false)
    f = open(filename)
    names = Any[]
    for l in eachline(f)
        push!(names, l[1:length(l)-erase_newline])        
    end
    close(f)
    return make_set ? Set(names) : names
end

#Takes the filename and returns each line in it
#works with flickr8k
function get_captions(img_names, filename; get_id=false, lower=true, erase_newline=true, base_dir="", cap_cnt=nothing)
    f = open(filename)
    captions = Any[]
    cap_cnts = (cap_cnt == nothing) ? nothing :  Dict()
    for l in eachline(f)
        imgvsrest = split(l, "#")
        img_name = string(base_dir, imgvsrest[1])
        idvscapt = split(imgvsrest[2], "\t")
        id, caption = idvscapt[1], string("<sw> ", idvscapt[2][1:length(idvscapt[2])-erase_newline], " <ew>")
        if lower; caption = lowercase(caption); end
        if last(split(img_name, "/")) in img_names && (cap_cnts == nothing || !haskey(cap_cnts, img_name) || cap_cnts[img_name] < cap_cnt)
            push!(captions, get_id ? (img_name, id, caption) : (img_name, caption))
            (cap_cnts != nothing) & (cap_cnts[img_name] = haskey(cap_cnts, img_name) ? cap_cnts[img_name]+1 : 1)
        end
    end
    close(f)
    return captions
end

caption_texts(captions; get_id=false) = map((c)->c[2+get_id], captions)
caption_image_names(captions) = map((c)->c[1], captions)

function create_dict(caption_texts; min_freq=5)
    dict = Dict()
    #start and end words
    #dict["<sw>"] = (1, Inf)
    #dict["<ew>"] = (2, Inf)
    
    #caption_texts = map(c2t, captions)
    largest_index = 0 #Largest accepted word index
    
    for t in caption_texts
        words = split(t)
        for w in words
            if haskey(dict, w)
                (index, freq) = dict[w]
                dict[w] = (index, freq+1)
            else
               dict[w] = (0, 1) 
            end
            if dict[w][2] == min_freq
                largest_index += 1
                dict[w] = (largest_index, min_freq)
            end
        end
    end

    #Crop the infrequent words
    for w in keys(dict)
        if dict[w][1] == 0
            delete!(dict, w)
        else
            dict[w] = dict[w][1]
        end
    end            
    return dict
end

function dictasarray(dict)
    dict_array = repmat([""], length(dict))    
    for k in keys(dict)
        dict_array[dict[k]] = k
    end
    return dict_array
end

function word2onehot(words, dict)
    ohv = zeros(Float32, length(dict), length(words))
    for i = 1:length(words)
        ohv[dict[words[i]], i] = 1.0
    end    
    #vect = Array(1:length(dict))
    return ohv
end

#Returns the largest possible caption in the dataset
function largest_caption(caption_texts, dict)
    largest_cap = 2
    for c in caption_texts
        w = filter((a)->a in keys(dict), split(c))
        largest_cap = max(length(w), largest_cap)
    end
    return largest_cap
end



function get_sequences(caption_texts, dict; max_caption=39)
    batch_size = length(caption_texts)
    caps = zeros(Float32, length(dict), max_caption, batch_size)
    masks = ones(UInt8, batch_size, max_caption)
    for b = 1:batch_size
        words = filter((w)->haskey(dict, w), split(caption_texts[b]))
        for i = 1:max_caption
            masks[b, i] = i <= length(words)
            wi = (masks[b,i] == 1) ? dict[words[i]] : dict["<ew>"]
            caps[wi, i, b] = 1.0
        end
    end
    return caps, masks    
end

function get_images(img_dsts)
    imgs = Any[]
    for i = 1:length(img_dsts)
        push!(imgs, mean_subtract(img2vec(load(img_dsts[i]))))
    end
    return imgs
end

function get_images(img_dsts, img_mean; unisize=true, dtype=Float32, img_size=(224,224,3))
    
    if unisize
        imgs = zeros(dtype, (img_size[1], img_size[2], img_size[3], length(img_dsts)))
        for i = 1:length(img_dsts)
            imgs[:, :, :, i] = img2vec(load(img_dsts[i])) .- img_mean
        end
        return imgs
    else                    
        imgs = Any[]
        for i = 1:length(img_dsts)
            push!(imgs, img2vec(load(img_dsts[i])) .- img_mean)
        end
        return imgs
    end        
end


    #DEATH CODE

    
#Load the image files and corresponding caption texts as a vector,
#applies mean subtraction while loading as it is a standard preprocession operation by default
#but this can be changed with mean_sub parameter
#Requires img_dists and caption_texts to be same size
function img_seq_pairs(img_dsts, caption_texts, dict; max_caption=39, mean_sub=true, img_cache=nothing, img_only = false, seq_only=false)

    img_only && seq_only && error("Assertion failed: At least one of seq_only and img_only should be false")
    #(length(caption_texts)!=length(img_dsts)) && error("Assertion failed: Different image and caption number.")
    batch_size = img_only ? length(img_dsts) : length(caption_texts)
    #load the images
    if !seq_only
        imgs = Any[]
        for i = 1:batch_size
            if img_cache != nothing && haskey(img_cache, img_dsts[i])
                imgs[:, :, :, i] = img_cache[img_dsts[i]]
            else
                imgs[:, :, :, i] = mean_sub ? mean_subtract(img2vec(load(img_dsts[i]))) : img2vec(load(img_dsts[i]))
            
                if img_cache != nothing
                    img_cache[img_dsts[i]] = imgs[:, :, :, i]
                end
            end                
        end
    end
    
    #load the captions
    if !img_only
        caps = zeros(Float32, length(dict), max_caption, batch_size)
        masks = ones(UInt8, batch_size, max_caption)
        for b = 1:batch_size
            words = filter((w)->haskey(dict, w), split(caption_texts[b]))
            for i = 1:max_caption
                masks[b, i] = i <= length(words)
                wi = (masks[b,i] == 1) ? dict[words[i]] : dict["<ew>"]
                caps[wi, i, b] = 1.0
            end
        end
    end
    
    if seq_only
        return caps, masks
    elseif img_only
        return imgs
    else        
        return imgs, caps, masks
    end
    
end




#UNUSED
function minibatcher(batch_size; training_images_file = "~/NIC/Flickr8k/text/Flickr_8k.trainImages.txt",
                     captions_file="~/NIC/Flickr8k/Flickr8k.token.txt", img_base_dir="~/NIC/Flickr8k/Flicker8k_Dataset_Prep224/",
                     shuffle=true, num_epochs=5, crop=(a)->a)
    start_index=1
    end_index=batch_size
    epoch = 1
    
    function next_batch()
        #old_start, old_end = start_index, end_index
        start_index = start_index + batch_size
        end_index = end_index + batch_size
        #New epoch
        if end_index >= length(captions)
            start_index, end_index = 1, batch_size
            if shuffle; shuffle!(captions); end
            epochs = epochs + 1
        end
        current_caps = captions[start_index:end_index]
        filenames = map((s) -> split(s, "#")[1], current_caps)
        
        #Populate target image data and labels
        #ygold = map((s) -> split(s, "\t")[2], current_caps)
        #image_batch = zeros(224, 224, 3, batch_size)
        #for i = 1:length(filenames)
        #    image_batch[:, :, :, i] = crop(img2vec(load(string(img_base_dir, filenames[i]))))
        #end
        return image_batch, ygold
        #return old_start, old_end        
    end

    function has_batch()
        return epoch <= num_epochs
    end    

    return next_batch, has_batch
end

function get_captions_junk(img_names, filename)
    d = Dict()
    f = open(filename)
    for l in eachline(f)
        id_cap = split(l, " ")
        words = id_cap[2:length(id_cap)]
        img_name = split(id_cap, "#")[1]
        if img_name in img_names
            if !haskey(d, img_name)
                d[img_name] = Any[]
            end
            push!(d[img_name], words)
        end
    end        
    close(f)
    return d
end
