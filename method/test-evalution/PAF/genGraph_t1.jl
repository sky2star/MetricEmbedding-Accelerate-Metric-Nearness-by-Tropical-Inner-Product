function genGraph_t1(n,p) 
    # Generate Type 1 graph: 
    # w(e) = 0 with probability p
    # w(e) = 1 with probability 1-p
    G = rand(n,n)
    G = triu(G)
    G += G'
    G[1:n+1:n*n] .= 0
    G[G.>p] .= 1
    G[G.<=p] .= 0
    
    return G
end