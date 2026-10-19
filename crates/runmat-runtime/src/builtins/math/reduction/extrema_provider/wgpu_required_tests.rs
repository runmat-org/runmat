//! Required-hardware Runtime extrema owner witnesses. Ordinary portable tests use an
//! in-process provider; these ignored tests must fail when no WGPU adapter exists.
use super::ExtremaKind;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostIntegerDataView, HostIntegerTensorView,
    HostLogicalTensorView, HostNumericDataView, HostNumericTensorView, IntegerElementType,
    ProviderPrecision, ThreadProviderGuard,
};
use runmat_value::{IntegerStorage, Tensor, Value};

fn selected_wgpu() -> &'static dyn AccelProvider {
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("required extrema WGPU qualification needs a live adapter");
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        let actual = match provider.precision() {
            ProviderPrecision::F32 => "f32",
            ProviderPrecision::F64 => "f64",
        };
        assert_eq!(
            requested.to_ascii_lowercase(),
            actual,
            "requested hardware precision was not selected"
        );
    }
    provider
}

fn host_tensor(value: &Value) -> Tensor {
    match value {
        gpu @ Value::GpuTensor(_) => {
            test_support::gather(gpu.clone()).expect("resident extrema download")
        }
        Value::Tensor(tensor) => tensor.clone(),
        other => panic!("expected nonscalar extrema tensor, got {other:?}"),
    }
}

fn evaluate_pair(kind: ExtremaKind, value: Value, rest: &[Value]) -> crate::execution::RootedValue {
    let arguments: Vec<Value> = std::iter::once(value).chain(rest.iter().cloned()).collect();
    let output = futures::executor::block_on(crate::call_builtin_async_with_outputs(
        kind.name(),
        &arguments,
        2,
    ))
    .expect("registered extrema evaluation");
    let Value::OutputList(outputs) = output.value() else {
        panic!("{}: expected two registered extrema outputs", kind.name());
    };
    assert_eq!(outputs.len(), 2, "two extrema outputs");
    output
}

fn output_pair(output: &crate::execution::RootedValue) -> (&Value, &Value) {
    let Value::OutputList(outputs) = output.value() else {
        panic!("expected extrema output list");
    };
    let [values, indices] = outputs.as_slice() else {
        panic!("expected two extrema outputs");
    };
    (values, indices)
}

#[test]
#[ignore = "requires a live WGPU adapter; run explicitly as required hardware qualification"]
fn actual_wgpu_min_max_default_and_dim_two_keep_values_and_double_indices() {
    let _serial = test_support::accel_test_lock();
    let provider = selected_wgpu();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let host_values = [3.0, 1.0, 2.0, 4.0];
    let input = match provider.precision() {
        ProviderPrecision::F32 => {
            let data = host_values.map(|value| value as f32);
            provider.upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&data),
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
        }
        ProviderPrecision::F64 => provider.upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&host_values),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        }),
    }
    .expect("typed WGPU upload");
    let empty = Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap());
    for (kind, controls, expected_values, expected_indices, expected_shape) in [
        (
            ExtremaKind::Min,
            vec![],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![1, 2],
        ),
        (
            ExtremaKind::Max,
            vec![],
            vec![3.0, 4.0],
            vec![1.0, 2.0],
            vec![1, 2],
        ),
        (
            ExtremaKind::Min,
            vec![empty.clone(), Value::Num(2.0)],
            vec![2.0, 1.0],
            vec![2.0, 1.0],
            vec![2, 1],
        ),
        (
            ExtremaKind::Max,
            vec![empty.clone(), Value::Num(2.0)],
            vec![3.0, 4.0],
            vec![1.0, 2.0],
            vec![2, 1],
        ),
    ] {
        let output = evaluate_pair(kind, Value::GpuTensor(input.clone()), &controls);
        let (values, indices) = output_pair(&output);
        let Value::GpuTensor(ref resident_values) = values else {
            panic!(
                "{}: supported WGPU dimensional values must remain resident",
                kind.name()
            );
        };
        assert_eq!(resident_values.shape(), expected_shape);
        assert_eq!(
            runmat_accelerate_api::handle_precision(resident_values),
            Some(provider.precision())
        );
        assert!(
            gpu_helpers::exact_provider_for_handle(resident_values).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert_eq!(host_tensor(values).materialize_f64(), expected_values);
        let indices = host_tensor(indices);
        assert_eq!(indices.shape(), expected_shape);
        assert_eq!(indices.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(indices.materialize_f64(), expected_indices);
        input.allocation_lease().validate_live().unwrap();
    }
}

#[test]
#[ignore = "requires a live WGPU adapter; run explicitly as required hardware qualification"]
fn actual_wgpu_min_max_keep_wide_u64_values_exact() {
    let _serial = test_support::accel_test_lock();
    let provider = selected_wgpu();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let data = [u64::MAX, 1_u64 << 63, 9, 7];
    let input = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&data),
            shape: &[2, 2],
        })
        .expect("exact WGPU uint64 upload");
    for (kind, expected, origins) in [
        (ExtremaKind::Min, vec![1_u64 << 63, 7], vec![2.0, 2.0]),
        (ExtremaKind::Max, vec![u64::MAX, 9], vec![1.0, 1.0]),
    ] {
        let output = evaluate_pair(kind, Value::GpuTensor(input.clone()), &[]);
        let (values, indices) = output_pair(&output);
        let Value::GpuTensor(ref resident_values) = values else {
            panic!("{}: native uint64 values must remain resident", kind.name());
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(resident_values),
            Some(IntegerElementType::U64)
        );
        assert_eq!(resident_values.shape(), &[1, 2]);
        assert!(
            gpu_helpers::exact_provider_for_handle(resident_values).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert_eq!(
            host_tensor(values),
            Tensor::new_integer(IntegerStorage::U64(expected), vec![1, 2]).unwrap()
        );
        let indices = host_tensor(indices);
        assert_eq!(indices.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(indices.materialize_f64(), origins);
        input.allocation_lease().validate_live().unwrap();
    }
}

#[test]
#[ignore = "requires a live WGPU adapter; run explicitly as required hardware qualification"]
fn actual_wgpu_logical_extrema_keep_double_values_and_origins() {
    let _serial = test_support::accel_test_lock();
    let provider = selected_wgpu();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 0, 0, 1],
            shape: &[2, 2],
        })
        .expect("actual WGPU logical upload");
    for (kind, expected_values, expected_indices) in [
        (ExtremaKind::Min, vec![0.0, 0.0], vec![2.0, 1.0]),
        (ExtremaKind::Max, vec![1.0, 1.0], vec![1.0, 2.0]),
    ] {
        let output = evaluate_pair(kind, Value::GpuTensor(input.clone()), &[]);
        let (values, indices) = output_pair(&output);
        let values = host_tensor(values);
        let indices = host_tensor(indices);
        assert_eq!(values.shape(), &[1, 2]);
        assert_eq!(values.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(values.materialize_f64(), expected_values);
        assert_eq!(indices.shape(), &[1, 2]);
        assert_eq!(indices.numeric_dtype(), runmat_value::NumericDType::F64);
        assert_eq!(indices.materialize_f64(), expected_indices);
        input.allocation_lease().validate_live().unwrap();
    }
}

#[test]
#[ignore = "requires a live WGPU adapter; run explicitly as required hardware qualification"]
fn actual_wgpu_nan_and_signed_zero_preserve_extrema_origins() {
    let _serial = test_support::accel_test_lock();
    let provider = selected_wgpu();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let data = [-0.0, 0.0, f64::NAN, f64::NAN];
    let input = match provider.precision() {
        ProviderPrecision::F32 => {
            let represented = data.map(|value| value as f32);
            provider.upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&represented),
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
        }
        ProviderPrecision::F64 => provider.upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&data),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        }),
    }
    .expect("actual WGPU signed-zero and NaN upload");
    for (kind, zero_bits, first_origin) in [
        (ExtremaKind::Min, (-0.0f64).to_bits(), 1.0),
        (ExtremaKind::Max, 0.0f64.to_bits(), 2.0),
    ] {
        let output = evaluate_pair(kind, Value::GpuTensor(input.clone()), &[]);
        let (values, indices) = output_pair(&output);
        assert!(matches!(&values, Value::GpuTensor(_)));
        let values = host_tensor(values).materialize_f64();
        assert_eq!(values[0].to_bits(), zero_bits, "{} zero sign", kind.name());
        assert!(values[1].is_nan(), "{} NaN propagation", kind.name());
        assert_eq!(
            host_tensor(indices).materialize_f64(),
            vec![first_origin, 1.0],
            "{} one-based origins",
            kind.name()
        );
        input.allocation_lease().validate_live().unwrap();
    }
}

#[test]
#[ignore = "requires a live WGPU adapter; run explicitly as required hardware qualification"]
fn actual_wgpu_explicit_all_and_vecdim_materialize_through_owning_provider() {
    let _serial = test_support::accel_test_lock();
    let provider = selected_wgpu();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let data = [3.0, 1.0, 2.0, 4.0];
    let input = match provider.precision() {
        ProviderPrecision::F32 => {
            let represented = data.map(|value| value as f32);
            provider.upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&represented),
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
        }
        ProviderPrecision::F64 => provider.upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&data),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        }),
    }
    .expect("actual WGPU explicit input upload")
    .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
    .expect("explicit input view");
    let host = match provider.precision() {
        ProviderPrecision::F32 => Value::Tensor(
            Tensor::from_f32(data.map(|value| value as f32).to_vec(), vec![2, 2]).unwrap(),
        ),
        ProviderPrecision::F64 => Value::Tensor(Tensor::new(data.to_vec(), vec![2, 2]).unwrap()),
    };
    let empty = Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap());
    let vecdim = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    for controls in [vec![empty.clone(), Value::from("all")], vec![empty, vecdim]] {
        for kind in [ExtremaKind::Min, ExtremaKind::Max] {
            let expected = evaluate_pair(kind, host.clone(), &controls);
            let output = evaluate_pair(kind, Value::GpuTensor(input.clone()), &controls);
            let (values, indices) = output_pair(&output);
            assert!(!matches!(&values, Value::GpuTensor(_)));
            assert!(!matches!(&indices, Value::GpuTensor(_)));
            assert_eq!(
                output.value(),
                expected.value(),
                "{} owning host route",
                kind.name()
            );
            input.allocation_lease().validate_live().unwrap();
        }
    }
}
