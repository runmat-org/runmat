using Random, Statistics, Printf

function main()
    Random.seed!(0)
    p, C, T = 128, 2048, 200
    mu = Float32(0.5); eps0 = Float32(1e-3)

    W = zeros(Float32, p, C)

    for _ in 1:T
        x = rand(Float32, p, C)
        d = vec(sum(x .* x, dims=1))
        y = vec(sum(x .* W, dims=1))
        e = d .- y
        nx = vec(sum(x .^ 2, dims=1)) .+ eps0
        scale = reshape(e ./ nx, 1, C) # 1×C broadcast over rows
        W .+= mu .* x .* scale
    end

    # mse against last step inputs
    x = rand(Float32, p, C) # regenerate to avoid undefined use; optional reuse
    d = vec(sum(x .* x, dims=1))
    mse = mean((d .- vec(sum(x .* W, dims=1))).^2)
    @printf("RESULT_ok MSE=%.6e\n", Float64(mse))
end

main()
