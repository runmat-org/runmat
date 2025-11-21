using Random, Statistics, Printf

function main()
    Random.seed!(0)
    M, T = 10_000_000, 256
    S0, mu, sigma = 100.0f0, 0.05f0, 0.20f0
    dt, K = 1.0f0 / 252.0f0, 100.0f0

    S = fill(S0, M, 1)
    sqrt_dt = sqrt(dt)
    drift = (mu - 0.5f0 * sigma * sigma) * dt
    scale = sigma * sqrt_dt

    for _ in 1:T
        Z = randn(Float32, M, 1)
        S .= S .* exp.(drift .+ scale .* Z)
    end

    payoff = max.(S .- K, 0.0f0)
    price = mean(payoff) * exp(-mu * T * dt)
    @printf("RESULT_ok PRICE=%.6f\n", Float64(price))
end

main()