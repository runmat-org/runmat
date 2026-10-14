use super::*;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    ProviderPrecision,
};

fn nonsquare_pool_adjoint_repeats_exact_native_seed_without_host_transfer(
    precision: ProviderPrecision,
) {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    assert_eq!(
        provider.precision(),
        precision,
        "set RUNMAT_WGPU_FORCE_PRECISION to the test precision; F64 requires SHADER_F64"
    );
    let x = (0..24).map(|index| index as f64).collect::<Vec<_>>();
    let dy = [0.6, -1.2, 0.3, 1.5, -0.9, 0.12, 2.1, -0.3];
    let input = match provider.precision() {
        ProviderPrecision::F32 => {
            let native = x.iter().map(|&value| value as f32).collect::<Vec<_>>();
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&native),
                    shape: &[2, 3, 4],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        }
        ProviderPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&x),
                shape: &[2, 3, 4],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
    };
    let upstream = match provider.precision() {
        ProviderPrecision::F32 => {
            let native = dy.map(|value| value as f32);
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&native),
                    shape: &[2, 1, 4],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap()
        }
        ProviderPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&dy),
                shape: &[2, 1, 4],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
    };
    let request = DeepLearningPool1DRequest { axis_zero_based: 1 };
    let before = provider.telemetry_snapshot();
    assert_eq!(
        provider
            .deep_learning_pool_1d_adjoint_capability_exec(
                &input,
                &[2, 3, 4],
                Some(&upstream),
                request,
            )
            .unwrap(),
        DeepLearningPool1DCapability::Ready,
    );
    assert_eq!(
        provider.telemetry_snapshot(),
        before,
        "admission must be pure"
    );
    let malformed = match provider.precision() {
        ProviderPrecision::F32 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.0_f32; 8]),
                shape: &[2, 4, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
        ProviderPrecision::F64 => provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[0.0_f64; 8]),
                shape: &[2, 4, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap(),
    };
    let before_invalid = provider.telemetry_snapshot();
    assert!(provider
        .deep_learning_pool_1d_adjoint_capability_exec(
            &input,
            &[2, 3, 4],
            Some(&malformed),
            request,
        )
        .is_err());
    assert_eq!(provider.telemetry_snapshot(), before_invalid);
    provider.reset_telemetry();
    let gradient = provider
        .deep_learning_pool_1d_adjoint_exec(&input, &upstream, request)
        .unwrap();
    assert_eq!(gradient.shape(), &[2, 3, 4]);
    assert_eq!(gradient.device_id(), input.device_id());
    assert_eq!(
        gradient.physical().element_type(),
        input.physical().element_type()
    );
    assert!(!gradient.shares_allocation_with(&input));
    assert!(!gradient.shares_allocation_with(&upstream));
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    let actual = match pollster::block_on(provider.download_numeric(&gradient))
        .unwrap()
        .data
    {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("expected native floating result, got {other:?}"),
    };
    assert_eq!(actual.len(), x.len());
    let tolerance = match provider.precision() {
        ProviderPrecision::F32 => 2e-7,
        ProviderPrecision::F64 => 1e-14,
    };
    for outer in 0..4 {
        for pooled in 0..3 {
            for inner in 0..2 {
                let index = inner + 2 * (pooled + 3 * outer);
                let expected = dy[inner + 2 * outer] / 3.0;
                assert!(
                    (actual[index] - expected).abs() < tolerance,
                    "dX[{index}]={} != {expected}",
                    actual[index]
                );
            }
        }
    }
    let source_after = match pollster::block_on(provider.download_numeric(&input))
        .unwrap()
        .data
    {
        HostNumericDataOwned::F32(values) => values.into_iter().map(f64::from).collect::<Vec<_>>(),
        HostNumericDataOwned::F64(values) => values,
        other => panic!("expected native floating source, got {other:?}"),
    };
    assert_eq!(source_after, x);
}

#[test]
#[ignore = "requires actual WGPU; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn nonsquare_pool_adjoint_f32() {
    nonsquare_pool_adjoint_repeats_exact_native_seed_without_host_transfer(ProviderPrecision::F32);
}

#[test]
#[ignore = "requires SHADER_F64 WGPU; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn nonsquare_pool_adjoint_f64() {
    nonsquare_pool_adjoint_repeats_exact_native_seed_without_host_transfer(ProviderPrecision::F64);
}

#[test]
#[ignore = "requires an actual WGPU adapter; run explicitly as a required-device gate"]
fn empty_observation_batch_pool_adjoint_keeps_native_shape() {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    let (input, upstream) = match provider.precision() {
        ProviderPrecision::F32 => (
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[]),
                    shape: &[0, 3, 4],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F32(&[]),
                    shape: &[0, 1, 4],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
        ),
        ProviderPrecision::F64 => (
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&[]),
                    shape: &[0, 3, 4],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
            provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::F64(&[]),
                    shape: &[0, 1, 4],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap(),
        ),
    };
    let gradient = provider
        .deep_learning_pool_1d_adjoint_exec(
            &input,
            &upstream,
            DeepLearningPool1DRequest { axis_zero_based: 1 },
        )
        .unwrap();
    assert_eq!(gradient.shape(), &[0, 3, 4]);
    match pollster::block_on(provider.download_numeric(&gradient))
        .unwrap()
        .data
    {
        HostNumericDataOwned::F32(values) => assert!(values.is_empty()),
        HostNumericDataOwned::F64(values) => assert!(values.is_empty()),
        other => panic!("expected native floating result, got {other:?}"),
    }
}
