#![cfg(feature = "wgpu")]
use runmat_accelerate_api::{
    AccelProvider, HostTensorView, MatmulEpilogue, MatmulOffset, ProviderPrecision,
};

#[tokio::test]
#[ignore = "requires a compatible actual WGPU device; run separately for each supported forced precision"]
async fn ordered_matmul_hardware_preserves_domains_orientation_and_diagonal() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required matrix hardware qualification needs a compatible adapter");
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        let actual = match provider.precision() {
            ProviderPrecision::F32 => "f32",
            ProviderPrecision::F64 => "f64",
        };
        assert_eq!(
            requested.to_ascii_lowercase(),
            actual,
            "required precision unavailable; must not count fallback as qualification"
        );
    }
    let identity = provider
        .upload(&HostTensorView {
            data: &[1., 0., 0., 1.],
            shape: &[2, 2],
        })
        .unwrap();
    let mut ep = MatmulEpilogue::noop();
    ep.affine.offset = Some(MatmulOffset::SubtractFrom(-0.));
    let zero = provider
        .upload(&HostTensorView {
            data: &[0.; 4],
            shape: &[2, 2],
        })
        .unwrap();
    let result = provider
        .matmul_epilogue(&zero, &identity, &ep)
        .await
        .unwrap();
    let host = provider.download(&result).await.unwrap();
    assert!(host.data.iter().all(|x| x.to_bits() == (-0.0f64).to_bits()));
    provider.free(&result).unwrap();
    provider.free(&zero).unwrap();

    let input = provider
        .upload(&HostTensorView {
            data: &[-2., 4., -4., 2.],
            shape: &[2, 2],
        })
        .unwrap();
    for (offset, expected) in [
        (MatmulOffset::Subtract(2.), [-4., 2., -6., 0.]),
        (MatmulOffset::SubtractFrom(2.), [4., -2., 6., 0.]),
    ] {
        ep = MatmulEpilogue::noop();
        ep.affine.offset = Some(offset);
        let result = provider
            .matmul_epilogue(&input, &identity, &ep)
            .await
            .unwrap();
        assert_eq!(provider.download(&result).await.unwrap().data, expected);
        provider.free(&result).unwrap();
    }
    ep = MatmulEpilogue::noop();
    ep.pow_exponent = Some(2.);
    let result = provider
        .matmul_epilogue(&input, &identity, &ep)
        .await
        .unwrap();
    for (actual, expected) in provider
        .download(&result)
        .await
        .unwrap()
        .data
        .into_iter()
        .zip([4., 16., 16., 4.])
    {
        assert!(
            (actual - expected).abs() <= expected * 1e-6,
            "negative integral power: {actual} vs {expected}"
        );
    }
    let diagonal = provider.diag_extract(&result, 0).unwrap();
    assert_eq!(diagonal.shape, [2, 1]);
    for value in provider.download(&diagonal).await.unwrap().data {
        assert!((value - 4.).abs() < 4e-6);
    }
    provider.free(&diagonal).unwrap();
    provider.free(&result).unwrap();
    provider.free(&input).unwrap();

    let input = provider
        .upload(&HostTensorView {
            data: &[f64::NAN, 1., 1., 1.],
            shape: &[2, 2],
        })
        .unwrap();
    ep = MatmulEpilogue::noop();
    ep.pow_exponent = Some(0.);
    let result = provider
        .matmul_epilogue(&input, &identity, &ep)
        .await
        .unwrap();
    let data = provider.download(&result).await.unwrap().data;
    assert!(data[0].is_nan() && data[2].is_nan());
    assert_eq!(data[1], 1.);
    assert_eq!(data[3], 1.);
    provider.free(&result).unwrap();
    provider.free(&input).unwrap();
    provider.free(&identity).unwrap();
}
