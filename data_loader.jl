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

#Takes the filenames and 
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
