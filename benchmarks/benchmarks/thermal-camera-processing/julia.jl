using Random, Statistics, Printf

function main()
    Random.seed!(0)
    B, H, W = 16, 1024, 1024

    raw = rand(Float32, B, H, W)

    dark   = 0.02f0 .+ 0.01f0 .* rand(Float32, H, W)
    ffc    = 0.98f0 .+ 0.04f0 .* rand(Float32, H, W)
    gain   = 1.50f0 .+ 0.50f0 .* rand(Float32, H, W)
    offset = -0.05f0 .+ 0.10f0 .* rand(Float32, H, W)

    lin = (raw .- dark) .* ffc
    radiance = lin .* gain .+ offset
    radiance = max.(radiance, 0.0f0)

    tempK = 273.15f0 .+ 80.0f0 .* log1p.(radiance)

    mean_temp = mean(tempK)
    @printf("RESULT_ok MEAN_TEMP=%.6f\n", Float64(mean_temp))
end

main()