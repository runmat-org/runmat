using Random, Statistics, Printf

function main()
    Random.seed!(0)
    M, T = 2_000_000, 4096
    alpha = Float32(0.98); beta = Float32(0.02)

    X = rand(Float32, M, T)
    Y = zeros(Float32, M, 1)

    for t in 1:T
        @inbounds Y .= alpha .* Y .+ beta .* X[:, t:t]
    end

    mean_y = mean(Y)
    @printf("RESULT_ok MEAN=%.6e\n", Float64(mean_y))
end

main()