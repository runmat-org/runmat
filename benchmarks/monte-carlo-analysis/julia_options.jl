using Statistics, Printf
const GPU = get(ENV, "MC_DEVICE", "cpu") == "gpu"
if GPU
    using Metal
    Metal.allowscalar(false)
end

function price_paths(z, mode)
    T = eltype(z)
    if mode == "loop"
        payoff = similar(z)
        for i in eachindex(z)
            payoff[i] = T(exp(-0.03)) * max(T(100) * exp(T(0.01) + T(0.2)*z[i]) - T(100), T(0))
        end
    elseif mode == "vectorized"
        payoff = T(exp(-0.03)) .* max.(T(100) .* exp.(T(0.01) .+ T(0.2) .* z) .- T(100), T(0))
    else
        error("MC_MODE must be loop or vectorized")
    end
    return mean(payoff), std(payoff) / sqrt(length(z))
end

function main()
    bytes = read(ENV["MC_FIXTURE"])
    z = collect(reinterpret(Float64, ltoh.(reinterpret(UInt64, bytes))))
    if get(ENV, "MC_PRECISION", "float64") == "float32"
        z = Float32.(z)
    end
    host_z = z
    roundtrip = get(ENV, "MC_TIMING", "resident") == "end-to-end"
    if GPU
        Metal.functional() || error("Metal GPU is unavailable")
        println("DEVICE gpu Metal")
        if !roundtrip
            z = MtlArray(host_z)
            Metal.synchronize()
        end
    end
    mode = get(ENV, "MC_MODE", "vectorized")
    repeats = parse(Int, get(ENV, "MC_REPEATS", "10"))
    warmups = parse(Int, get(ENV, "MC_WARMUPS", "3"))
    for rep in 1:(warmups + repeats)
        GPU && Metal.synchronize()
        start = time_ns()
        if GPU && roundtrip
            z = MtlArray(host_z)
        end
        price, stderr = price_paths(z, mode)
        GPU && Metal.synchronize()
        elapsed = (time_ns() - start) / 1e6
        @printf("SAMPLE rep=%d ms=%.12g price=%.17g stderr=%.17g\n", rep, elapsed, price, stderr)
    end
end

main()
