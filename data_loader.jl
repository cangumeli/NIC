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
function get_captions(img_names, filename)
    f = open(filename)
    captions = Any[]
    for l in eachline(f)
        img_name = split(l, "#")[1]
        if img_name in img_names
            push!(captions, l)
        end
    end
    close(f)
    return captions
end

function create_dict(captions; c2t=(a)->split(a, "\t")[2], min_freq=5)
    dict = Dict()
    #start and end words
    dict["<sw>"] = (1, Inf)
    dict["<ew>"] = (2, Inf)
    
    caption_texts = map(c2t, captions)
    largest_index = 2 #Largest accepted word index
    
    for t in caption_texts
        words = map((s)->lowercase(s), split(t))
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


function word2onehot(word, dict)
    vect = Array(1:length(dict))
    return vect .== dict[word]
end

#retuns a tupple of the form ((data, caption), epoch)
#requires batchsize | # of captions
function minibatcher(batch_size, captions, img_base_dir; shuffle=true, num_epochs=5, crop=(a)->a)
    start_index=1
    end_index=batch_size
    epochs = 1
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
        ygold = map((s) -> split(s, "\t")[2], current_caps)
        image_batch = zeros(224, 224, 3, batch_size)
        for i = 1:length(filenames)
            image_batch[:, :, :, i] = crop(img2vec(load(string(img_base_dir, filenames[i]))))
        end
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
