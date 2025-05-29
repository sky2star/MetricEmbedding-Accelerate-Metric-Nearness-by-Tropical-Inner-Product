function compute_nmse(X, D)
    # compute the normalized squared error, i.e. ||X-D||^2 / ||D||^2
    nmse = norm(X-D)^2 / norm(D)^2
    return nmse
end