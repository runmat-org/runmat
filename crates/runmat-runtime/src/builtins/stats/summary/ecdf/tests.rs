use super::*;
use futures::executor::block_on;
use runmat_builtins::{
    builtin_function_by_name, BuiltinParamType, CDFPLOT_DESCRIPTOR, CDFPLOT_EXTENSIONS,
    CDFPLOT_INTEGER_CAPABILITIES, ECDF_EXTENSIONS,
};
use runmat_value::{IntegerStorage, LogicalArray};

fn canonical_entry(name: &str) -> &'static runmat_builtins::BuiltinCatalogEntry {
    let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name)
        .expect("canonical empirical identity");
    let bindings = crate::builtin::runtime_builtin_bindings_by_name(name);
    assert_eq!(bindings.len(), 1, "one canonical implementation binding");
    assert_eq!(bindings[0].identity.builtin, entry.identity);
    assert_eq!(bindings[0].identity.variant, "default");
    assert!(
        builtin_function_by_name(name).is_none(),
        "no legacy authority"
    );
    entry
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn numeric_ecdf_remains_available_without_presentation() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(vec![1.0, 2.0, 3.0])]),
    ))
    .expect("numeric ECDF output");
    assert!(matches!(result, Value::OutputList(values) if values.len() == 2));
    drop(_outputs);

    let _statement = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(vec![1.0, 2.0, 3.0])]),
    ))
    .expect_err("presentation unavailable");
    assert!(error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<crate::plotting_hooks::PlottingUnavailable>())
        .is_some());
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let logical = Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![2, 1]).unwrap());
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![logical]),
    ))
    .expect_err("valid logical statement still requires presentation");
    assert!(error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<crate::plotting_hooks::PlottingUnavailable>())
        .is_some());
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn cdfplot_valid_sample_reports_typed_missing_presentation_before_download() {
    use std::error::Error;

    let error = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(vec![1.0, 2.0, 3.0])]),
    ))
    .expect_err("presentation unavailable");
    assert!(error
        .source()
        .and_then(|source| source.downcast_ref::<crate::plotting_hooks::PlottingUnavailable>())
        .is_some());
}

fn tensor(data: Vec<f64>) -> Value {
    Value::Tensor(Tensor::new(data.clone(), vec![data.len(), 1]).unwrap())
}

fn int_tensor(storage: IntegerStorage, len: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![len, 1]).unwrap();
    Value::Tensor(tensor)
}

fn all_integer_storages(values: (i64, u64)) -> [IntegerStorage; 8] {
    let (signed, unsigned) = values;
    [
        IntegerStorage::I8(vec![signed as i8]),
        IntegerStorage::I16(vec![signed as i16]),
        IntegerStorage::I32(vec![signed as i32]),
        IntegerStorage::I64(vec![signed]),
        IntegerStorage::U8(vec![unsigned as u8]),
        IntegerStorage::U16(vec![unsigned as u16]),
        IntegerStorage::U32(vec![unsigned as u32]),
        IntegerStorage::U64(vec![unsigned]),
    ]
}

fn output_tensors(value: Value) -> Vec<Tensor> {
    let Value::OutputList(values) = value else {
        panic!("expected output list");
    };
    values
        .try_into_owned()
        .expect("unique output list")
        .into_iter()
        .map(|value| match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        })
        .collect()
}

fn matrix(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

#[test]
fn ecdf_numeric_parsers_read_admitted_typed_integer_storage() {
    let exact = (1_u64 << 53) - 1;
    assert_eq!(
        numeric_vector(
            &int_tensor(IntegerStorage::U64(vec![exact, exact - 1]), 2),
            ECDF_OWNER,
        )
        .unwrap(),
        vec![exact as f64, (exact - 1) as f64]
    );
    assert_eq!(
        scalar_number(&int_tensor(IntegerStorage::U64(vec![exact]), 1)).unwrap(),
        exact as f64
    );
}

#[test]
fn ecdf_accepts_typed_integer_data_frequency_and_censoring() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), 4),
            Value::from("Frequency"),
            int_tensor(IntegerStorage::U16(vec![2, 1, 1, 1]), 4),
            Value::from("Censoring"),
            int_tensor(IntegerStorage::I8(vec![0, 1, 0, 0]), 4),
        ]),
    ))
    .unwrap();
    let outputs = output_tensors(value);
    assert_eq!(outputs[1].materialize_f64(), vec![1.0, 1.0, 3.0, 4.0]);
    assert!((outputs[0].materialize_f64()[1] - 0.4).abs() < 1.0e-12);
    assert!((outputs[0].materialize_f64()[2] - 0.7).abs() < 1.0e-12);
}

#[test]
fn invalid_non_axes_forms_and_control_shapes_match_catalog_diagnostics() {
    let sample = || tensor(vec![1.0, 2.0]);
    assert!(block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![sample(), sample()])
    ))
    .is_err());
    assert!(block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![sample(), sample()])
    ))
    .is_err());
    assert!(block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            sample(),
            Value::from("Alpha"),
            Value::Bool(true),
        ])
    ))
    .is_err());
    assert!(block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            sample(),
            Value::from("Alpha"),
            Value::Complex(0.2, 0.0),
        ])
    ))
    .is_err());
    assert!(block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0]),
            Value::from("Frequency"),
            Value::Tensor(Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap()),
        ])
    ))
    .is_err());
    assert!(block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            sample(),
            Value::from("Censoring"),
            Value::ComplexTensor(
                runmat_value::ComplexTensor::new(vec![(0.0, 0.0), (0.0, 0.0)], vec![2, 1]).unwrap(),
            ),
        ])
    ))
    .is_err());
}

#[test]
fn ecdf_alpha_rejects_typed_integer_boundary_value() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let err = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            int_tensor(IntegerStorage::I16(vec![1, 2, 3]), 3),
            Value::from("Alpha"),
            int_tensor(IntegerStorage::U8(vec![1]), 1),
        ]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:ecdf:InvalidArgument"));
    assert!(err.message().contains("open interval"), "{}", err.message());
}

#[test]
fn ecdf_integer_roles_follow_strict_compatibility_mode() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer_y = int_tensor(IntegerStorage::I16(vec![1, 2]), 2);
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![integer_y]),
    ))
    .expect_err("integer y must gate");
    assert_eq!(
        error.identifier(),
        ECDF_INTEGER_Y_EXTENSION.error_identifier
    );

    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0]),
            Value::from("Frequency"),
            int_tensor(IntegerStorage::U16(vec![1, 1]), 2),
        ]),
    ))
    .expect_err("integer Frequency must gate");
    assert_eq!(
        error.identifier(),
        ECDF_INTEGER_FREQUENCY_EXTENSION.error_identifier
    );

    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0]),
            Value::from("Censoring"),
            int_tensor(IntegerStorage::I8(vec![0, 1]), 2),
        ]),
    ))
    .expect_err("integer Censoring must gate");
    assert_eq!(
        error.identifier(),
        ECDF_INTEGER_CENSORING_EXTENSION.error_identifier
    );
}

#[test]
fn ecdf_classifies_resident_integer_y_before_provider_access() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::I16,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
            handle.clone(),
        )]),
    ))
    .expect_err("resident integer y must gate before gather");
    assert_eq!(
        error.identifier(),
        ECDF_INTEGER_Y_EXTENSION.error_identifier
    );
}

#[cfg(feature = "plot-core")]
#[test]
fn cdfplot_dispatch_preserves_residency_until_builtin_preflight() {
    assert_eq!(CDFPLOT_GPU_SPEC.residency, ResidencyPolicy::NewHandle);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered resident fixture")
            .handle,
    );
    let prepared = block_on(runmat_accelerate::prepare_builtin_args(
        "cdfplot",
        &[resident],
    ))
    .expect("dispatcher must retain resident argument");
    assert!(matches!(prepared.as_slice(), [Value::GpuTensor(_)]));
}

#[test]
fn ecdf_admits_all_integer_y_classes_and_rejects_wide_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for storage in all_integer_storages((2, 2)) {
        let output = block_on(ecdf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(storage, 1)]),
        ))
        .expect("small integer y");
        assert!(matches!(output, Value::Tensor(_)));
    }
    let wide = int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), 1);
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![wide]),
    ))
    .expect_err("wide y must reject");
    assert!(
        error.message().contains("exactly representable as double"),
        "{}",
        error.message()
    );
}

#[test]
fn ecdf_accepts_aligned_wide_integer_at_the_exact_binary64_boundary() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let aligned = 1_u64 << 54;
    let result = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::U64(vec![aligned, aligned + 4]),
            2,
        )]),
    ))
    .expect("aligned wide integers are exactly representable");
    let output = output_tensors(result);
    assert_eq!(
        output[1].materialize_f64(),
        vec![aligned as f64, aligned as f64, (aligned + 4) as f64]
    );
}

#[test]
fn ecdf_rejects_fractional_frequency_counts() {
    let error = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0]),
            Value::from("Frequency"),
            tensor(vec![1.5, 1.0]),
        ]),
    ))
    .expect_err("fractional Frequency must reject");
    assert!(error.message().contains("integer counts"));
}

#[test]
fn ecdf_descriptor_exposes_role_specific_integer_policy() {
    let builtin = canonical_entry(ECDF_NAME);
    assert_eq!(builtin.extensions, &ECDF_EXTENSIONS);
    assert_eq!(builtin.integer_capabilities.len(), 3);
}

#[test]
fn ecdf_resident_input_restores_numeric_outputs_to_its_owner() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
            .expect("resident input");
        let result = block_on(ecdf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        ))
        .expect("resident ecdf");
        let Value::OutputList(outputs) = result else {
            panic!("expected output list");
        };
        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .all(|value| matches!(value, Value::GpuTensor(_))));
    });
}

#[test]
fn ecdf_returns_unique_sorted_points_with_initial_zero() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(vec![
            3.0, 1.0, 2.0, 2.0,
        ])]),
    ))
    .unwrap();
    let outputs = output_tensors(value);
    assert_eq!(outputs[0].materialize_f64(), vec![0.0, 0.25, 0.75, 1.0]);
    assert_eq!(outputs[1].materialize_f64(), vec![1.0, 1.0, 2.0, 3.0]);
}

#[test]
fn ecdf_logical_vector_preserves_shape_and_numeric_steps() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let sample = Value::LogicalArray(
        LogicalArray::new(vec![0, 1, 1, 0], vec![4, 1]).expect("logical vector"),
    );
    let outputs = output_tensors(
        block_on(ecdf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![sample]),
        ))
        .expect("logical ecdf"),
    );
    assert_eq!(outputs[0].materialize_f64(), vec![0.0, 0.5, 1.0]);
    assert_eq!(outputs[1].materialize_f64(), vec![0.0, 0.0, 1.0]);
    assert_eq!(outputs[0].shape(), [3, 1]);
    assert_eq!(outputs[1].shape(), [3, 1]);
}

#[test]
fn ecdf_logical_sample_geometry_matches_numeric_sample_admission() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    for (data, shape) in [(vec![], vec![0, 1]), (vec![1; 6], vec![2, 3])] {
        let sample = Value::LogicalArray(LogicalArray::new(data, shape).unwrap());
        let error = block_on(ecdf_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![sample]),
        ))
        .expect_err("invalid logical shape");
        assert_eq!(error.identifier(), Some("RunMat:ecdf:InvalidArgument"));
    }
}

#[test]
fn ecdf_supports_frequency_survivor_and_bounds_outputs() {
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let value = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0, 3.0]),
            Value::from("Frequency"),
            tensor(vec![2.0, 1.0, 1.0]),
            Value::from("Function"),
            Value::from("survivor"),
            Value::from("Alpha"),
            Value::Num(0.1),
        ]),
    ))
    .unwrap();
    let outputs = output_tensors(value);
    assert_eq!(outputs[0].materialize_f64(), vec![1.0, 0.5, 0.25, 0.0]);
    assert_eq!(outputs[1].materialize_f64(), vec![1.0, 1.0, 2.0, 3.0]);
    assert_eq!(outputs[2].shape(), [4, 1]);
    assert_eq!(outputs[3].shape(), [4, 1]);
}

#[test]
fn ecdf_supports_right_censoring_with_kaplan_meier_steps() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0]),
            Value::from("Censoring"),
            tensor(vec![0.0, 1.0, 0.0, 0.0]),
        ]),
    ))
    .unwrap();
    let outputs = output_tensors(value);
    assert_eq!(outputs[1].materialize_f64(), vec![1.0, 1.0, 3.0, 4.0]);
    assert!((outputs[0].materialize_f64()[1] - 0.25).abs() < 1.0e-12);
    assert!((outputs[0].materialize_f64()[2] - 0.625).abs() < 1.0e-12);
    assert!((outputs[0].materialize_f64()[3] - 1.0).abs() < 1.0e-12);
}

#[test]
fn ecdf_right_censoring_starts_at_first_event_time() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0]),
            Value::from("Censoring"),
            tensor(vec![1.0, 0.0, 0.0, 0.0]),
        ]),
    ))
    .unwrap();
    let outputs = output_tensors(value);
    assert_eq!(outputs[1].materialize_f64(), vec![2.0, 2.0, 3.0, 4.0]);
    assert!((outputs[0].materialize_f64()[1] - (1.0 / 3.0)).abs() < 1.0e-12);
}

#[test]
fn ecdf_cumulative_hazard_uses_nelson_aalen_steps() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0, 3.0]),
            Value::from("Function"),
            Value::from("cumulative hazard"),
        ]),
    ))
    .unwrap();
    let outputs = output_tensors(value);
    assert_eq!(outputs[1].materialize_f64(), vec![1.0, 1.0, 2.0, 3.0]);
    let expected = [
        0.0,
        1.0 / 3.0,
        1.0 / 3.0 + 1.0 / 2.0,
        1.0 / 3.0 + 1.0 / 2.0 + 1.0,
    ];
    for (actual, expected) in outputs[0].materialize_f64().iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
    assert!(outputs[0]
        .materialize_f64()
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn ecdf_rejects_non_vector_non_interval_matrix() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let err = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![matrix(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            3,
        )]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:ecdf:InvalidArgument"));
}

#[test]
fn ecdf_supports_left_censoring_without_right_censoring() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(ecdf_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            tensor(vec![1.0, 2.0, 3.0, 4.0]),
            Value::from("Censoring"),
            tensor(vec![-1.0, 0.0, 0.0, 0.0]),
        ]),
    ))
    .unwrap();
    let outputs = output_tensors(value);
    assert_eq!(outputs[1].materialize_f64(), vec![2.0, 2.0, 3.0, 4.0]);
    assert!(outputs[0]
        .materialize_f64()
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
}

#[cfg(feature = "plot-core")]
#[test]
fn cdfplot_returns_handle_and_stats() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let _ = crate::builtins::plotting::clear_figure(None);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![tensor(vec![1.0, 2.0, 3.0])]),
    ))
    .unwrap();
    let Value::OutputList(values) = value else {
        panic!("expected output list");
    };
    assert!(matches!(values[0], Value::Num(handle) if handle > 0.0));
    let Value::Struct(stats) = &values[1] else {
        panic!("expected stats struct");
    };
    assert!(matches!(stats.fields.get("min"), Some(Value::Num(1.0))));
    assert!(matches!(stats.fields.get("max"), Some(Value::Num(3.0))));
    assert!(matches!(stats.fields.get("mean"), Some(Value::Num(2.0))));
    assert!(matches!(stats.fields.get("median"), Some(Value::Num(2.0))));
}

#[cfg(feature = "plot-core")]
#[test]
fn cdfplot_rejects_nonvector_nd_data_before_mutating_graphics() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]).unwrap());
    let error = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![data]),
    ))
    .expect_err("N-D data is not a vector");
    assert_eq!(error.identifier(), Some("RunMat:cdfplot:InvalidArgument"));
    assert!(error.message().contains("expected a vector"), "{error}");
}

#[test]
fn cdfplot_descriptor_classifies_integer_and_runtime_extensions() {
    let builtin = canonical_entry(CDFPLOT_NAME);
    assert_eq!(builtin.extensions, &CDFPLOT_EXTENSIONS);
    assert_eq!(
        builtin
            .integer_capabilities
            .iter()
            .map(|capability| capability.form)
            .collect::<Vec<_>>(),
        CDFPLOT_INTEGER_CAPABILITIES
            .iter()
            .map(|capability| capability.form)
            .collect::<Vec<_>>()
    );
    assert_eq!(builtin.integer_capabilities[0].inputs[0].classes.len(), 8);
    assert_eq!(CDFPLOT_DESCRIPTOR.signatures[0].inputs[0].name, "x");
    assert_eq!(
        CDFPLOT_DESCRIPTOR.signatures[0].inputs[0].ty,
        BuiltinParamType::NumericArray
    );
}

#[cfg(feature = "plot-core")]
#[test]
fn cdfplot_admits_all_integer_classes_at_explicit_double_boundary() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    for storage in all_integer_storages((-3, 3)) {
        crate::builtins::plotting::reset_plot_state();
        let value = block_on(cdfplot_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(storage, 1)]),
        ))
        .expect("integer cdfplot");
        let Value::OutputList(outputs) = value else {
            panic!("expected output list");
        };
        assert!(matches!(outputs[0], Value::Num(handle) if handle > 0.0));
        let Value::Struct(stats) = &outputs[1] else {
            panic!("expected stats structure");
        };
        assert!(matches!(stats.fields.get("mean"), Some(Value::Num(value)) if value.abs() == 3.0));
    }
    crate::builtins::plotting::reset_plot_state();
    let logical = Value::LogicalArray(
        LogicalArray::new(vec![0, 1], vec![2, 1]).expect("logical sample data"),
    );
    let value = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![logical]),
    ))
    .expect("logical cdfplot");
    let Value::OutputList(outputs) = value else {
        panic!("expected output list");
    };
    let Value::Struct(stats) = &outputs[1] else {
        panic!("expected stats structure");
    };
    assert!(matches!(stats.fields.get("mean"), Some(Value::Num(0.5))));
}

#[cfg(feature = "plot-core")]
#[test]
fn cdfplot_strict_mode_rejects_integer_logical_and_gpu_extensions() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::I16(vec![1]),
            1,
        )]),
    ))
    .expect_err("integer input must be gated");
    assert_eq!(
        integer.identifier(),
        CDFPLOT_INTEGER_INPUT_EXTENSION.error_identifier
    );
    let logical = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
    ))
    .expect_err("logical input must be gated");
    assert_eq!(
        logical.identifier(),
        CDFPLOT_LOGICAL_INPUT_EXTENSION.error_identifier
    );
    let single =
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2, 1]).expect("single sample data"));
    let documented = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![single]),
    ))
    .expect("documented single input");
    assert!(matches!(documented, Value::Num(handle) if handle > 0.0));

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let integer_value = Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![2, 1])
            .expect("integer input");
        let integer_handle =
            crate::builtins::common::gpu_helpers::upload_tensor(provider, &integer_value)
                .expect("resident integer input");
        let integer_gpu = block_on(cdfplot_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                integer_handle,
            )]),
        ))
        .expect_err("resident integer type must be gated before gather");
        assert_eq!(
            integer_gpu.identifier(),
            CDFPLOT_INTEGER_INPUT_EXTENSION.error_identifier
        );
        let value = Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("double input");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &value)
            .expect("resident input");
        let gpu = block_on(cdfplot_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        ))
        .expect_err("gpu input must be gated before gather");
        assert_eq!(
            gpu.identifier(),
            CDFPLOT_GPU_INPUT_EXTENSION.error_identifier
        );
    });
}

#[cfg(feature = "plot-core")]
#[test]
fn cdfplot_rejects_wide_integer_before_lossy_plotting() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = block_on(cdfplot_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![int_tensor(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            1,
        )]),
    ))
    .expect_err("wide integer must not be rounded");
    assert!(
        error.message().contains("exact binary64 integer range"),
        "{}",
        error.message()
    );
}

#[cfg(feature = "plot-core")]
#[test]
fn cdfplot_resident_integer_fallback_returns_host_outputs() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let value = Tensor::new_integer(IntegerStorage::U32(vec![1, 3]), vec![2, 1])
            .expect("integer input");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &value)
            .expect("resident integer input");
        let result = block_on(cdfplot_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        ))
        .expect("resident integer cdfplot");
        let Value::OutputList(outputs) = result else {
            panic!("expected host outputs");
        };
        assert!(matches!(outputs[0], Value::Num(handle) if handle > 0.0));
        let Value::Struct(stats) = &outputs[1] else {
            panic!("expected stats structure");
        };
        assert!(matches!(stats.fields.get("mean"), Some(Value::Num(2.0))));
    });
}

#[test]
#[cfg(all(feature = "plot-core", feature = "wgpu"))]
fn cdfplot_wgpu_integer_fallback_returns_host_outputs_for_all_classes() {
    let _plot_guard = crate::builtins::plotting::lock_plot_test_context();
    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    for storage in all_integer_storages((2, 2)) {
        crate::builtins::plotting::reset_plot_state();
        let value = Tensor::new_integer(storage, vec![1, 1]).expect("integer input");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &value)
            .expect("resident integer input");
        let result = block_on(cdfplot_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle,
            )]),
        ))
        .expect("wgpu integer cdfplot");
        let Value::OutputList(outputs) = result else {
            panic!("expected host outputs");
        };
        assert!(matches!(outputs[0], Value::Num(handle) if handle > 0.0));
        let Value::Struct(stats) = &outputs[1] else {
            panic!("expected stats structure");
        };
        assert!(matches!(stats.fields.get("mean"), Some(Value::Num(2.0))));
    }
}
