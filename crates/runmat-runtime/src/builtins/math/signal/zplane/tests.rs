use super::presentation::zplane_limits;
use super::*;
use futures::executor::block_on;
use runmat_builtins::builtin_function_by_name;
use runmat_value::IntValue;
use runmat_value::IntegerStorage;

fn row(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).unwrap())
}

fn col(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
}

#[test]
fn descriptor_is_registered() {
    let entry = runmat_builtins::builtin_catalog_entry_by_name(BUILTIN_NAME).expect("catalog");
    assert!(builtin_function_by_name(BUILTIN_NAME).is_none());
    assert_eq!(entry.bindings.len(), 1);
    assert_eq!(entry.bindings[0].variant, "default");
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(BUILTIN_NAME);
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].identity,
        entry.binding_identity(&entry.bindings[0])
    );
    assert!(entry
        .descriptor
        .signatures
        .iter()
        .any(|sig| sig.label == "zplane(b, a)"));
}

#[test]
fn row_vectors_are_filter_coefficients() {
    let data = block_on(parse_zero_pole_data(
        crate::builtins::common::test_support::builtin_arguments(vec![
            row(&[1.0]),
            row(&[1.0, -0.8]),
        ]),
    ))
    .expect("zero-pole data");
    assert!(data.zeros.is_empty());
    assert_eq!(data.poles.len(), 1);
    assert!((data.poles[0].re - 0.8).abs() < 1e-10);
    assert!(data.poles[0].im.abs() < 1e-10);
}

#[test]
fn column_vectors_are_explicit_zero_pole_locations() {
    let data = block_on(parse_zero_pole_data(
        crate::builtins::common::test_support::builtin_arguments(vec![
            col(&[0.2, 0.4]),
            col(&[0.8]),
        ]),
    ))
    .expect("zero-pole data");
    assert_eq!(data.zeros.len(), 2);
    assert_eq!(data.poles.len(), 1);
    assert!((data.zeros[0].re - 0.2).abs() < 1e-12);
    assert!((data.poles[0].re - 0.8).abs() < 1e-12);
}

#[test]
fn sos_rows_expand_to_zeros_and_poles() {
    let sos = Value::Tensor(Tensor::new(vec![1.0, 1.0, 0.0, 1.0, -0.8, 0.0], vec![1, 6]).unwrap());
    let data = block_on(parse_zero_pole_data(
        crate::builtins::common::test_support::builtin_arguments(vec![sos]),
    ))
    .expect("sos");
    assert_eq!(data.zeros.len(), 2);
    assert_eq!(data.poles.len(), 2);
    assert!(data.poles.iter().any(|z| (z.re - 0.8).abs() < 1e-10));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn zplane_sos_dense_rank_zero_rejects_and_rank_one_is_a_row() {
    let rank_zero = Tensor::new(vec![1.0], vec![]).expect("rank-zero SOS");
    let error = block_on(evaluate(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rank_zero)]),
    ))
    .expect_err("rank-zero SOS has no valid matrix element count");
    assert_eq!(error.identifier(), ZPLANE_ERROR_INVALID_ARGUMENT.identifier);

    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let rank_one =
        Tensor::new(vec![1.0, 1.0, 0.0, 1.0, -0.8, 0.0], vec![6]).expect("rank-one SOS row");
    let Value::Tensor(handles) = block_on(evaluate(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rank_one)]),
    ))
    .expect("rank-one SOS is a six-column row") else {
        panic!("expected graphics handle tensor");
    };
    // Three guide lines plus both nonempty zero and pole scatter plots.
    assert_eq!(handles.shape(), [1, 5]);
}

#[test]
fn sos_reads_typed_integer_storage_exactly() {
    let tensor = Tensor::new_integer(IntegerStorage::I16(vec![1, 1, 0, 1, -1, 0]), vec![1, 6])
        .expect("integer sos");

    let data = block_on(parse_zero_pole_data(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(tensor)]),
    ))
    .expect("sos");

    assert_eq!(data.zeros.len(), 2);
    assert_eq!(data.poles.len(), 2);
    assert!(data
        .poles
        .iter()
        .any(|z| (z.re - 1.0).abs() < 1e-10 && z.im.abs() < 1e-10));
}

#[test]
fn sos_rejects_rank_greater_than_two() {
    let sos =
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 0.0); 12], vec![1, 6, 2]).unwrap());
    let err = block_on(parse_zero_pole_data(
        crate::builtins::common::test_support::builtin_arguments(vec![sos]),
    ))
    .expect_err("rank-3 sos rejected");
    assert!(err.message().contains("2-D matrix"));
}

#[test]
fn malformed_resident_sos_gain_and_vector_metadata_precede_download() {
    use crate::builtins::common::{gpu_helpers, test_support};

    test_support::with_test_provider(|provider| {
        let bad_sos = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0; 6], vec![2, 3]).expect("malformed SOS"),
        )
        .expect("resident SOS");
        let bad_gain = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("nonscalar gain"),
        )
        .expect("resident gain");
        let bad_vector = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0; 4], vec![2, 2]).expect("matrix vector"),
        )
        .expect("resident vector");
        for args in [
            vec![Value::GpuTensor(bad_sos.clone())],
            vec![
                col(&[0.2, 0.4]),
                col(&[0.8]),
                Value::GpuTensor(bad_gain.clone()),
            ],
            vec![Value::GpuTensor(bad_vector.clone()), row(&[1.0, -0.8])],
        ] {
            provider.reset_telemetry();
            let error = block_on(evaluate(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect_err("known-invalid resident metadata");
            assert_eq!(error.identifier(), ZPLANE_ERROR_INVALID_ARGUMENT.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
        for handle in [bad_sos, bad_gain, bad_vector] {
            assert!(handle.allocation_lease().validate_live().is_ok());
            handle.release().expect("release malformed input");
        }
    });
}

#[test]
fn leading_resident_complex_integer_rejects_before_exactness_readback() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};

    test_support::with_test_provider(|provider| {
        // Canonical host numeric upload intentionally refuses complex integer
        // computation. A fixture-owned native allocation exercises the
        // authoritative resident descriptor that must still be rejected here.
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let scalar = owner
            .allocate_zeroed(
                vec![1, 1],
                NumericElementType::U64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .expect("fixture resident complex integer scalar")
            .handle;
        let sos = owner
            .allocate_zeroed(
                vec![1, 6],
                NumericElementType::U64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .expect("fixture resident complex integer SOS")
            .handle;
        for args in [
            vec![Value::GpuTensor(sos.clone())],
            vec![Value::GpuTensor(sos.clone()), row(&[1.0, -0.8])],
            vec![
                Value::GpuTensor(scalar.clone()),
                row(&[1.0]),
                row(&[1.0, -0.8]),
            ],
        ] {
            provider.reset_telemetry();
            let error = block_on(evaluate(
                crate::builtins::common::test_support::builtin_arguments(args),
            ))
            .expect_err("unsupported complex integer");
            assert_eq!(error.identifier(), ZPLANE_ERROR_INVALID_ARGUMENT.identifier);
            assert!(error.message().contains("complex integer"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
        for handle in [scalar, sos] {
            assert!(handle.allocation_lease().validate_live().is_ok());
            handle.release().expect("release original input");
        }
    });
}

#[test]
fn explicit_zpk_accepts_complex_scalar_gain() {
    let data = block_on(parse_zero_pole_data(
        crate::builtins::common::test_support::builtin_arguments(vec![
            col(&[0.2, 0.4]),
            col(&[0.8]),
            Value::Complex(1.0, 0.25),
        ]),
    ))
    .expect("complex gain accepted");
    assert_eq!(data.zeros.len(), 2);
    assert_eq!(data.poles.len(), 1);
}

#[test]
fn explicit_zpk_gain_reads_typed_integer_storage_exactly() {
    let gain = Tensor::new_integer(IntegerStorage::U8(vec![1]), vec![1, 1]).expect("integer gain");

    let data = block_on(parse_zero_pole_data(
        crate::builtins::common::test_support::builtin_arguments(vec![
            col(&[0.2, 0.4]),
            col(&[0.8]),
            Value::Tensor(gain),
        ]),
    ))
    .expect("integer gain accepted");

    assert_eq!(data.zeros.len(), 2);
    assert_eq!(data.poles.len(), 1);
}

#[test]
fn zplane_limits_include_unit_circle_and_points() {
    let limits = zplane_limits(&ZeroPoleData {
        zeros: vec![Complex::new(2.0, -0.5)],
        poles: vec![Complex::new(0.0, -3.0)],
    });
    assert!(limits.0 < -3.0);
    assert!(limits.1 > 3.0);
    assert!((limits.0 + limits.1).abs() < 1e-12);
}

#[test]
fn evaluate_returns_graphics_handles_without_interactive_backend() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let value = block_on(evaluate(
        crate::builtins::common::test_support::builtin_arguments(vec![
            row(&[1.0]),
            row(&[1.0, -0.8]),
        ]),
    ))
    .expect("zplane render");
    let Value::Tensor(handles) = value else {
        panic!("expected handle tensor");
    };
    assert_eq!(handles.shape(), [1, 4]);
    assert!(handles
        .materialize_f64()
        .iter()
        .all(|handle| handle.is_finite()));
}

#[test]
fn evaluate_accepts_leading_axes_for_coefficient_and_sos_forms() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let axes = crate::builtins::plotting::axes::axes_builtin(vec![]).expect("axes");
    for data in [
        vec![row(&[1.0]), row(&[1.0, -0.8])],
        vec![row(&[1.0, 0.0, 0.0, 1.0, -0.8, 0.0])],
        vec![col(&[0.2]), col(&[0.8]), Value::Num(1.0)],
    ] {
        let mut args = vec![Value::Num(axes)];
        args.extend(data);
        let result = block_on(evaluate(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect("targeted zplane");
        let Value::Tensor(handles) = result else {
            panic!("assigned targeted plot returns handles");
        };
        assert_eq!(handles.shape()[0], 1);
        assert!(handles.shape()[1] >= 4);
    }
}

#[test]
fn evaluate_honors_zero_requested_outputs() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let _out_guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let value = block_on(evaluate(
        crate::builtins::common::test_support::builtin_arguments(vec![
            row(&[1.0]),
            row(&[1.0, -0.8]),
        ]),
    ))
    .expect("zplane render");
    let Value::OutputList(values) = value else {
        panic!("expected output list");
    };
    assert!(values.is_empty());
}

#[test]
fn evaluate_preserves_first_handle_output_and_pads_later_requested_slots() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let _out_guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(evaluate(
        crate::builtins::common::test_support::builtin_arguments(vec![
            row(&[1.0]),
            row(&[1.0, -0.8]),
        ]),
    ))
    .expect("zplane render");
    let Value::OutputList(values) = value else {
        panic!("expected two requested outputs");
    };
    assert_eq!(values.len(), 2);
    let Value::Tensor(handles) = &values[0] else {
        panic!("first output must contain graphics handles");
    };
    assert_eq!(handles.shape(), [1, 4]);
    assert!(matches!(&values[1], Value::Num(n) if *n == 0.0));
}

#[test]
fn zplane_integer_extension_rejects_before_plot_state() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let coefficients = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![1, 2]).expect("coefficients"),
    );
    let error = block_on(evaluate(
        crate::builtins::common::test_support::builtin_arguments(vec![
            coefficients,
            row(&[1.0, -0.8]),
        ]),
    ))
    .expect_err("strict gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:ZplaneNonfloatingInputExtension")
    );
}

#[test]
fn wide_integer_vector_and_sos_reject_before_lossy_root_conversion() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let wide = (1_u64 << 53) + 1;
    for args in [
        vec![Value::Int(IntValue::U64(wide)), row(&[1.0, -0.8])],
        vec![
            Value::Int(IntValue::U64(wide)),
            row(&[1.0]),
            row(&[1.0, -0.8]),
        ],
        vec![Value::Tensor(
            Tensor::new_integer(IntegerStorage::U64(vec![1, 0, 0, 1, wide, 0]), vec![1, 6])
                .expect("SOS integer tensor"),
        )],
    ] {
        let error = block_on(evaluate(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .expect_err("inexact integer rejected");
        assert_eq!(error.identifier(), ZPLANE_ERROR_INVALID_ARGUMENT.identifier);
        assert!(error.message().contains("exactly representable"));
    }
}

#[test]
fn wide_resident_integer_rejects_with_original_allocation_live() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let wide = (1_u64 << 53) + 1;
        let input = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(std::slice::from_ref(&wide)),
                shape: &[1, 1],
            })
            .expect("resident integer");
        let error = block_on(evaluate(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(input.clone()),
                row(&[1.0, -0.8]),
            ]),
        ))
        .expect_err("inexact resident integer rejected");
        assert_eq!(error.identifier(), ZPLANE_ERROR_INVALID_ARGUMENT.identifier);
        assert!(input.allocation_lease().validate_live().is_ok());
        input.release().expect("release original input");
    });
}

#[test]
fn vector_validation_wraps_and_retains_shared_cause() {
    let error = block_on(vector_input(
        "numerator",
        Value::String("bad".to_string().into()),
    ))
    .expect_err("invalid coefficient");
    assert_eq!(error.identifier(), ZPLANE_ERROR_INVALID_ARGUMENT.identifier);
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn sos_and_gain_operational_failures_keep_zplane_owner_and_typed_source() {
    let cause = crate::build_runtime_error("resident download cancelled")
        .with_identifier("RunMat:gather")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    let error = map_internal(cause);
    assert_eq!(error.identifier(), ZPLANE_ERROR_INTERNAL.identifier);
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
    let source = std::error::Error::source(&error).expect("typed gather source");
    assert!(source.to_string().contains("resident download cancelled"));
}

#[test]
fn resident_zpk_gain_is_admitted_and_source_remains_live() {
    use crate::builtins::common::gpu_helpers;
    use crate::builtins::common::test_support;

    test_support::with_test_provider(|provider| {
        let gain = Tensor::new(vec![2.0], vec![1, 1]).expect("gain");
        let handle = gpu_helpers::upload_tensor(provider, &gain).expect("resident gain");
        let data = block_on(parse_zero_pole_data(
            crate::builtins::common::test_support::builtin_arguments(vec![
                col(&[0.2]),
                col(&[0.8]),
                Value::GpuTensor(handle.clone()),
            ]),
        ))
        .expect("resident ZPK gain");
        assert_eq!(data.zeros.len(), 1);
        assert_eq!(data.poles.len(), 1);
        assert!(handle.allocation_lease().validate_live().is_ok());
        handle.release().expect("release original gain");
    });
}

#[test]
fn inexact_resident_zpk_gain_is_rejected_before_materialization() {
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let wide = (1_u64 << 53) + 1;
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(std::slice::from_ref(&wide)),
                shape: &[1, 1],
            })
            .expect("resident integer gain");
        let error = block_on(evaluate(
            crate::builtins::common::test_support::builtin_arguments(vec![
                col(&[0.2, 0.4]),
                col(&[0.8]),
                Value::GpuTensor(handle.clone()),
            ]),
        ))
        .expect_err("inexact gain rejected before plot");
        assert_eq!(error.identifier(), ZPLANE_ERROR_INVALID_ARGUMENT.identifier);
        assert!(error.message().contains("exactly representable"));
        assert!(handle.allocation_lease().validate_live().is_ok());
        handle.release().expect("release gain");
    });
}
