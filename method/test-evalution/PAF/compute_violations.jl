function compute_violations(D, epsilon = 1e-10)
    # Compute the number of violated triangle inequalities
    (n,n) = size(D)
    count = 0
    # epsilon = 1e-10
    for i = 1:n-1
        for j = i+1:n
            num = sum(D[i,j]-epsilon .> D[:,i]+D[:,j])
            count += num
        end
    end

    return count
end
            