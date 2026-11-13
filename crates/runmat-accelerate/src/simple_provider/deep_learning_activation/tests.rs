use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
};

#[test]
fn native_single_forward_activations_preserve_precision_and_owner_without_download() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[-2.0, -0.0, 3.0, 1.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    provider.reset_telemetry();
    let relu = block_on(provider.activation_relu(&input)).unwrap();
    let elu = block_on(provider.activation_elu(&input, 1.0)).unwrap();
    let softmax = block_on(provider.activation_softmax_rows(&input)).unwrap();
    for handle in [&relu, &elu, &softmax] {
        assert_eq!(handle.physical().element_type(), NumericElementType::F32);
        assert!(handle.allocation_lease().validate_live().is_ok());
    }
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(
        block_on(provider.download_numeric(&relu)).unwrap().data,
        HostNumericDataOwned::F32(vec![0.0, 0.0, 3.0, 1.0])
    );
    let HostNumericDataOwned::F32(elu_values) =
        block_on(provider.download_numeric(&elu)).unwrap().data
    else {
        panic!("single ELU must retain F32")
    };
    assert!((elu_values[0] - ((-2.0_f32).exp() - 1.0)).abs() < 1e-6);
    assert_eq!(&elu_values[1..], &[0.0, 3.0, 1.0]);
    assert!(elu_values[1].is_sign_negative());
    let HostNumericDataOwned::F32(probabilities) =
        block_on(provider.download_numeric(&softmax)).unwrap().data
    else {
        panic!("single softmax must retain F32")
    };
    // Column-major rows are [-2, 3] and [0, 1], independently normalized.
    let first = (-5.0_f32).exp() / ((-5.0_f32).exp() + 1.0);
    let second = 1.0 / (1.0 + 1.0_f32.exp());
    assert!((probabilities[0] - first).abs() < 1e-6);
    assert!((probabilities[1] - second).abs() < 1e-6);
    assert!((probabilities[2] - (1.0 - first)).abs() < 1e-6);
    assert!((probabilities[3] - (1.0 - second)).abs() < 1e-6);
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[test]
fn relu_preserves_nan_and_signed_zero_policy_in_native_classes() {
    let provider = InProcessProvider::new();
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[f32::NAN, -0.0, -2.0, 3.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let double = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[f64::NAN, -0.0, -2.0, 3.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    provider.reset_telemetry();
    let single_output = block_on(provider.activation_relu(&single)).unwrap();
    let double_output = block_on(provider.activation_relu(&double)).unwrap();
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    let HostNumericDataOwned::F32(single_values) =
        block_on(provider.download_numeric(&single_output))
            .unwrap()
            .data
    else {
        panic!("ReLU changed single precision")
    };
    let HostNumericDataOwned::F64(double_values) =
        block_on(provider.download_numeric(&double_output))
            .unwrap()
            .data
    else {
        panic!("ReLU changed double precision")
    };
    assert!(single_values[0].is_nan() && double_values[0].is_nan());
    assert_eq!(&single_values[1..], &[0.0, 0.0, 3.0]);
    assert_eq!(&double_values[1..], &[0.0, 0.0, 3.0]);
    assert!(!single_values[1].is_sign_negative() && !double_values[1].is_sign_negative());
    assert_eq!(
        single_output.physical().element_type(),
        NumericElementType::F32
    );
    assert_eq!(
        double_output.physical().element_type(),
        NumericElementType::F64
    );
}

#[test]
fn single_elu_rejects_unrepresentable_alpha_before_publishing_output() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[-1.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let upstream = input.clone();
    let alpha = f64::MAX;
    assert!(!provider.activation_adjoint_supported(
        ProviderActivationAdjoint::Elu { alpha },
        runmat_accelerate_api::ProviderPrecision::F32,
    ));
    assert!(block_on(provider.activation_elu(&input, alpha)).is_err());
    assert!(block_on(provider.activation_adjoint(
        &input,
        &upstream,
        ProviderActivationAdjoint::Elu { alpha },
    ))
    .is_err());
    assert!(input.allocation_lease().validate_live().is_ok());
}

#[test]
fn native_softmax_propagates_nonfinite_rows_instead_of_provider_only_error() {
    let provider = InProcessProvider::new();
    let shape = [2, 2];
    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[f32::NAN, f32::INFINITY, 1.0, 0.0]),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let double = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[f64::NAN, f64::INFINITY, 1.0, 0.0]),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    provider.reset_telemetry();
    for (input, class) in [
        (&single, NumericElementType::F32),
        (&double, NumericElementType::F64),
    ] {
        let result = block_on(provider.activation_softmax_rows(input)).unwrap();
        assert_eq!(result.physical().element_type(), class);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let values = block_on(provider.download(&result)).unwrap();
        assert_eq!(values.shape, shape);
        assert!(values.data.iter().all(|value| value.is_nan()));
        provider.reset_telemetry();
    }
}

#[test]
fn native_single_and_double_adjoints_preserve_class_and_ownership() {
    let provider = InProcessProvider::new();
    let x32 = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[-2., 0., 3.]),
            shape: &[1, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let dy32 = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[2., 4., 5.]),
            shape: &[1, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    provider.reset_telemetry();
    let relu = block_on(provider.activation_adjoint(&x32, &dy32, ProviderActivationAdjoint::Relu))
        .unwrap();
    assert_eq!(relu.physical().element_type(), NumericElementType::F32);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(
        block_on(provider.download_numeric(&relu)).unwrap().data,
        HostNumericDataOwned::F32(vec![0., 0., 5.])
    );
    let elu = block_on(provider.activation_adjoint(
        &x32,
        &dy32,
        ProviderActivationAdjoint::Elu { alpha: 1.25 },
    ))
    .unwrap();
    let HostNumericDataOwned::F32(values) = block_on(provider.download_numeric(&elu)).unwrap().data
    else {
        panic!("native single ELU adjoint expected")
    };
    assert!((values[0] - 2.5 * (-2.0_f32).exp()).abs() < 1e-6);
    assert_eq!(&values[1..], &[4., 5.]);
    let x64 = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[-1., 2.]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let dy64 = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[3., 4.]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let result = block_on(provider.activation_adjoint(
        &x64,
        &dy64,
        ProviderActivationAdjoint::Elu { alpha: 2. },
    ))
    .unwrap();
    assert_eq!(result.physical().element_type(), NumericElementType::F64);
    let HostNumericDataOwned::F64(values) =
        block_on(provider.download_numeric(&result)).unwrap().data
    else {
        panic!("native double ELU adjoint expected")
    };
    assert!((values[0] - 6. * (-1.0_f64).exp()).abs() < 1e-12);
    assert_eq!(values[1], 4.);
    assert!(x32.allocation_lease().validate_live().is_ok());
    assert!(dy64.allocation_lease().validate_live().is_ok());
}

#[test]
fn adjoint_rejects_foreign_and_mixed_precision_without_publishing_output() {
    let provider = InProcessProvider::new();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let mixed = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let foreign = InProcessProvider::new()
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(
        block_on(provider.activation_adjoint(&input, &mixed, ProviderActivationAdjoint::Relu))
            .is_err()
    );
    assert!(block_on(provider.activation_adjoint(
        &input,
        &foreign,
        ProviderActivationAdjoint::Relu
    ))
    .is_err());
    assert!(input.allocation_lease().validate_live().is_ok());
}
