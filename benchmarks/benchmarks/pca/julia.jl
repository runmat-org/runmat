using Random, LinearAlgebra, Statistics, Printf

function main()
    Random.seed!(0)
    n, d, k, iters = 200_000, 1024, 8, 15

    A = rand(Float32, n, d)
    mu = mean(A, dims=1)
    A .-= mu
    G = (A' * A) / Float32(n - 1)

    Q = rand(Float32, d, k)
    Q = Matrix(qr(Q, Thin=true).Q)

    for _ in 1:iters
        Q = G * Q
        Q = Matrix(qr(Q, Thin=true).Q)
    end

    Lambda = diag(Q' * G * Q)
    explained = Float64.(Lambda) ./ sum(diag(G))
    @printf("RESULT_ok EXPLAINED1=%.4f TOPK_SUM=%.6e\n", explained[1], sum(Float64.(Lambda)))
end

main()

