function BregmanOrig_v1(D, N)
    D_ini = copy(D)
    (n,n) = size(D)
    g = SimpleWeightedGraph(D)
    Z = Dict()
    Zt = spzeros(n,n)
    
    niter = 0
    Niter = N
    nmse = zeros(1, Niter+1)
    violation = zeros(1, Niter+1)
    violation[1,1] = compute_violations(D_ini)
    update = zeros(1, Niter+1)
    sec = zeros(1, Niter+1)
    count = 0
    # maxD = 1
    # tol = 1e-4

    while (niter < Niter)  # (maxD > tol) 
        start_time = time()
        t2 = @elapsed g, Z, Zt, count = BregmanOrig_step(n, copy(g), copy(Z), copy(Zt), copy(count))

        D = copy(g.weights)
        niter += 1
        nmse[1, niter+1] = compute_nmse(D, D_ini)
        violation[1, niter+1] = compute_violations(D)
        update[1, niter+1] = count
        t = time() - start_time
        sec[1, niter+1] = sec[1, niter] + t
        println("Saved Iter: ", niter)
        println("nmse: ",nmse[1, niter+1])
        println("violation: ",violation[1, niter+1])
        println("sec: ",sec[1, niter+1])
        println("sec2",t)
    end
    D = g.weights
    return (D, nmse, violation, update, sec)
end


function BregmanOrig_step(n, g, Z, Zt, count)
    for p in keys(Z)
        z = Z[p]
        t = length(p)
        u = p[1]
        v = p[t]
        d = -1*g.weights[u,v]
        for i = 1:t-1
            d = d + g.weights[p[i], p[i+1]]
        end
        c = min(d/t,z)
        for i = 1:t-1
            g.weights[p[i],p[i+1]] -= c
            g.weights[p[i+1],p[i]] -= c
            count += 1
        end
        g.weights[u,v] += c
        g.weights[v,u] += c
        count += 1
        if z == c
            delete!(Z,p)
        else
            Z[p] -= c
        end
    end

    for i = 1:n
        for j = 1:i-1
            c = min(g.weights[j,i] - 1e-14, Zt[j,i])
            g.weights[j,i] -= c
            g.weights[i,j] -= c
            Zt[j,i] -= c
            Zt[i,j] -= c      
            count += 2
        end
    end

    FS = Graphs.floyd_warshall_shortest_paths(g)
    U = FS.dists
    P = enumerate_paths2(FS)
    # maxD = 0

    for i = 1:n
        for j = 1:i-1
            if g.weights[j,i] - U[j,i] > 0  
                p = P[j][i] 
                t = length(p)
                u = p[1]
                v = p[t]
                d = -1*g.weights[u,v]
                for k = 1:t-1
                    d = d + g.weights[p[k], p[k+1]]
                end
                if d < 0
                    c = d/t
                    for k = 1:t-1
                        g.weights[p[k],p[k+1]] -= c
                        g.weights[p[k+1],p[k]] -= c
                        count += 1
                    end
                    g.weights[p[1],p[t]] += c
                    g.weights[p[t],p[1]] += c
                    count += 1
                    if haskey(Z,p)
                        Z[p] = Z[p] - c
                    else
                        Z[p] = -1*c
                    end
                    # if abs(d) > maxD
                    #     maxD = abs(d)
                    # end
                end
            end   
        end
    end

    return (g, Z, Zt, count)
end


function enumerate_paths2(s)
    P = Array{Any,1}(undef,size(s.parents, 1))
    for v = 1:size(s.parents, 1)
        P[v] = Graphs.enumerate_paths(s, v)
    end
    
    return P
end
