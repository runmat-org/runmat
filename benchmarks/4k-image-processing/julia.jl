using Random, Statistics, Printf

function main()
    Random.seed!(0)
    B, H, W = 16, 2160, 3840
    gain = Float32(1.0123); bias = Float32(-0.02); gamma = Float32(1.8); eps0 = Float32(1e-6)

    imgs = rand(Float32, B, H, W)
    mu = mean(imgs, dims=(2,3))
    sigma = sqrt.(mean((imgs .- mu).^2, dims=(2,3)) .+ eps0)

    out = ((imgs .- mu) ./ sigma) .* gain .+ bias
    out = out .^ gamma
    mse = mean((out .- imgs) .^ 2)
    @printf("RESULT_ok MSE=%.6e\n", Float64(mse))
end

main()