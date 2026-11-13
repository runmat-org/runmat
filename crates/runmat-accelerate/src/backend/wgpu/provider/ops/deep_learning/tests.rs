use crate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelProvider, HostTensorView, ProviderAdamUpdateRequest, ProviderCrossentropyMode,
    ProviderCrossentropyRequest,
};

#[test]
fn adam_update_wgpu_returns_resident_outputs() {
    let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
        return;
    };

    let shape = [1usize, 3usize];
    let parameters = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &shape,
        })
        .expect("upload parameters");
    let gradient = provider
        .upload(&HostTensorView {
            data: &[0.1, -0.2, 0.3],
            shape: &shape,
        })
        .expect("upload gradient");

    let result = provider
        .adam_update(&ProviderAdamUpdateRequest {
            parameters: &parameters,
            gradient: &gradient,
            average_grad: None,
            average_sq_grad: None,
            iteration: 1,
            learn_rate: 0.01,
            gradient_decay_factor: 0.9,
            squared_gradient_decay_factor: 0.999,
            epsilon: 1.0e-8,
        })
        .expect("adam update");

    let updated =
        pollster::block_on(provider.download(&result.parameters)).expect("download params");
    let avg = pollster::block_on(provider.download(&result.average_grad)).expect("download avg");
    let avg_sq =
        pollster::block_on(provider.download(&result.average_sq_grad)).expect("download avg sq");

    assert_eq!(updated.shape, shape);
    assert_eq!(avg.shape, shape);
    assert_eq!(avg_sq.shape, shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
        runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
    };
    let expected_updated = [0.990000001, 2.0099999995, 2.9900000003333334];
    let expected_avg = [0.01, -0.02, 0.03];
    let expected_avg_sq = [0.00001, 0.00004, 0.00009];
    for idx in 0..shape[1] {
        assert!(
            (updated.data[idx] - expected_updated[idx]).abs() < tol,
            "updated lane {idx}: got {}, expected {}",
            updated.data[idx],
            expected_updated[idx]
        );
        assert!(
            (avg.data[idx] - expected_avg[idx]).abs() < tol,
            "avg lane {idx}: got {}, expected {}",
            avg.data[idx],
            expected_avg[idx]
        );
        assert!(
            (avg_sq.data[idx] - expected_avg_sq[idx]).abs() < tol,
            "avg sq lane {idx}: got {}, expected {}",
            avg_sq.data[idx],
            expected_avg_sq[idx]
        );
    }

    for handle in [
        parameters,
        gradient,
        result.parameters,
        result.average_grad,
        result.average_sq_grad,
    ] {
        handle.release().unwrap();
    }
}

#[test]
#[ignore = "requires an actual WGPU device; run explicitly for resident activation evidence"]
fn relu_wgpu_preserves_nan_and_zero_branch() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU device required for ReLU activation test");
    let shape = [1usize, 4usize];
    let input = provider
        .upload(&HostTensorView {
            data: &[f64::NAN, -0.0, -2.0, 3.0],
            shape: &shape,
        })
        .expect("upload input");
    let before = provider.telemetry_snapshot().download_bytes;
    let output = pollster::block_on(provider.activation_relu(&input)).expect("resident ReLU");
    assert_eq!(provider.telemetry_snapshot().download_bytes, before);
    let actual = pollster::block_on(provider.download(&output)).expect("download output");
    assert_eq!(actual.shape, shape);
    assert!(actual.data[0].is_nan());
    assert_eq!(&actual.data[1..], &[0.0, 0.0, 3.0]);
    assert!(!actual.data[1].is_sign_negative());
    let upstream = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &shape,
        })
        .expect("upload upstream gradient");
    let before_reverse = provider.telemetry_snapshot().download_bytes;
    let adjoint = pollster::block_on(provider.activation_adjoint(
        &input,
        &upstream,
        runmat_accelerate_api::ProviderActivationAdjoint::Relu,
    ))
    .expect("resident ReLU adjoint");
    assert_eq!(provider.telemetry_snapshot().download_bytes, before_reverse);
    let gradient = pollster::block_on(provider.download(&adjoint)).expect("download adjoint");
    assert!(gradient.data[0].is_nan());
    assert_eq!(&gradient.data[1..], &[0.0, 0.0, 4.0]);
    input.release().unwrap();
    output.release().unwrap();
    upstream.release().unwrap();
    adjoint.release().unwrap();
}

#[test]
#[ignore = "requires an actual WGPU device; run explicitly for resident activation evidence"]
fn elu_wgpu_matches_cpu_branch_semantics() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU device required for ELU activation test");
    let shape = [2usize, 2usize];
    let input = provider
        .upload(&HostTensorView {
            data: &[-2.0, -0.0, 3.0, f64::NAN],
            shape: &shape,
        })
        .expect("upload input");
    let output = pollster::block_on(provider.activation_elu(&input, 1.25)).expect("resident ELU");
    let actual = pollster::block_on(provider.download(&output)).expect("download output");
    let expected = [1.25 * ((-2.0_f64).exp() - 1.0), 0.0, 3.0, f64::NAN];
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
        runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
    };
    assert_eq!(actual.shape, shape);
    for (actual, expected) in actual.data.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance || (actual.is_nan() && expected.is_nan()),
            "got {actual}, expected {expected}"
        );
    }
    assert!(actual.data[1].is_sign_negative());
    input.release().unwrap();
    output.release().unwrap();
}

#[test]
#[ignore = "requires an actual WGPU device; run explicitly for resident softmax evidence"]
fn softmax_rows_wgpu_is_stable_and_resident() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU device required for softmax test");
    let shape = [2usize, 3usize];
    let input = provider
        .upload(&HostTensorView {
            // Column-major rows: [1000, 0] and [1001, -1] and [999, 1].
            data: &[1000.0, 0.0, 1001.0, -1.0, 999.0, 1.0],
            shape: &shape,
        })
        .expect("upload input");
    let output =
        pollster::block_on(provider.activation_softmax_rows(&input)).expect("resident softmax");
    let actual = pollster::block_on(provider.download(&output)).expect("download output");
    let tolerance = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
        runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
    };
    let e = std::f64::consts::E;
    let expected = [
        1.0 / (1.0 + e + 1.0 / e),
        1.0 / (1.0 + 1.0 / e + e),
        e / (1.0 + e + 1.0 / e),
        (1.0 / e) / (1.0 + 1.0 / e + e),
        (1.0 / e) / (1.0 + e + 1.0 / e),
        e / (1.0 + 1.0 / e + e),
    ];
    assert_eq!(actual.shape, shape);
    for (actual, expected) in actual.data.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "got {actual}, expected {expected}"
        );
    }
    input.release().unwrap();
    output.release().unwrap();
}

#[test]
#[ignore = "requires an actual WGPU device; run explicitly for resident softmax evidence"]
fn softmax_rows_wgpu_propagates_undefined_nonfinite_normalization() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU device required for nonfinite softmax test");
    let shape = [1usize, 2usize];
    let input = provider
        .upload(&HostTensorView {
            data: &[f64::INFINITY, 0.0],
            shape: &shape,
        })
        .expect("upload input");
    let before = provider.telemetry_snapshot().download_bytes;
    let output = pollster::block_on(provider.activation_softmax_rows(&input))
        .expect("nonfinite row produces a resident NaN result");
    assert_eq!(provider.telemetry_snapshot().download_bytes, before);
    let actual = pollster::block_on(provider.download(&output)).expect("download output");
    assert!(actual.data.iter().all(|value| value.is_nan()));
    input.release().unwrap();
    output.release().unwrap();
}

#[test]
fn crossentropy_wgpu_returns_resident_loss_terms() {
    let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
        return;
    };

    let shape = [1usize, 2usize];
    let predictions = provider
        .upload(&HostTensorView {
            data: &[0.8, 0.2],
            shape: &shape,
        })
        .expect("upload predictions");
    let targets = provider
        .upload(&HostTensorView {
            data: &[1.0, 0.0],
            shape: &shape,
        })
        .expect("upload targets");
    let weights = provider
        .upload(&HostTensorView {
            data: &[2.0, 3.0],
            shape: &shape,
        })
        .expect("upload weights");
    let mask = provider
        .upload(&HostTensorView {
            data: &[1.0, 0.0],
            shape: &shape,
        })
        .expect("upload mask");

    let result = provider
        .crossentropy_terms(&ProviderCrossentropyRequest {
            predictions: &predictions,
            targets: &targets,
            weights: Some(&weights),
            mask: Some(&mask),
            mode: ProviderCrossentropyMode::MultiLabel,
        })
        .expect("crossentropy terms");
    let losses = pollster::block_on(provider.download(&result.losses)).expect("download");
    assert_eq!(losses.shape, shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1.0e-10,
        runmat_accelerate_api::ProviderPrecision::F32 => 1.0e-5,
    };
    let expected = [-2.0 * 0.8_f64.ln(), 0.0];
    for (idx, actual) in losses.data.iter().enumerate() {
        assert!(
            (*actual - expected[idx]).abs() < tol,
            "loss lane {idx}: got {actual}, expected {}",
            expected[idx]
        );
    }

    for handle in [predictions, targets, weights, mask, result.losses] {
        handle.release().unwrap();
    }
}
