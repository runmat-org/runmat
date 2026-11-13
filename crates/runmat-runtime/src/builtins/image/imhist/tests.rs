use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use crate::context::host_presentation::tests::Witness as PresentationWitness;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, NumericDType};
use std::rc::Rc;

fn call(image: Value, rest: Vec<Value>, outputs: Option<usize>) -> Value {
    let _guard = outputs
        .map(|count| crate::output_count::push_output_count(Some(count)))
        .transpose()
        .expect("output-count scope");
    block_on(imhist_builtin(
        None,
        image,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
    .expect("imhist")
}

fn tensor(data: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> Tensor {
    Tensor::new_with_dtype(data, shape, dtype).unwrap()
}

#[test]
fn canonical_boundary_has_no_legacy_gpu_or_fusion_spec_shadow() {
    use crate::builtins::common::spec::{
        builtin_fusion_specs, builtin_gpu_specs, canonical_fusion_declaration,
    };
    let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(NAME)
        .expect("compiled canonical imhist");
    assert_eq!(
        entry.placement.preparation,
        runmat_builtins::BuiltinOperandPreparation::RuntimeOwned
    );
    assert_eq!(
        entry.placement.fusion,
        runmat_builtins::BuiltinFusionPolicy::Boundary
    );
    assert!(!builtin_gpu_specs().any(|spec| spec.name == NAME));
    assert!(!builtin_fusion_specs().any(|spec| spec.name == NAME));
    assert!(canonical_fusion_declaration(entry.binding_identity(&entry.bindings[0])).is_none());
}

#[test]
fn uint8_grayscale_default_bins_count_exact_intensities() {
    let image = tensor(vec![0.0, 1.0, 1.0, 255.0], vec![2, 2], NumericDType::U8);
    let Value::Tensor(counts) = call(Value::Tensor(image), vec![], None) else {
        panic!("expected counts tensor");
    };
    assert_eq!(counts.shape(), &[256, 1]);
    assert_eq!(counts.materialize_f64()[0], 1.0);
    assert_eq!(counts.materialize_f64()[1], 2.0);
    assert_eq!(counts.materialize_f64()[255], 1.0);
}

#[test]
fn exact_uint8_storage_uses_integer_image_range() {
    let image = Tensor::new_integer(IntegerStorage::U8(vec![0, 1, 1, 2, 2, 2]), vec![2, 3])
        .expect("uint8 image");
    let Value::Tensor(counts) = call(Value::Tensor(image), vec![], None) else {
        panic!("expected counts tensor");
    };
    assert_eq!(counts.shape(), &[256, 1]);
    assert_eq!(counts.materialize_f64()[0], 1.0);
    assert_eq!(counts.materialize_f64()[1], 2.0);
    assert_eq!(counts.materialize_f64()[2], 3.0);
}

#[test]
fn two_outputs_return_counts_and_bin_locations_as_columns() {
    let image = tensor(vec![0.0, 0.5, 1.0, 1.0], vec![2, 2], NumericDType::F64);
    let Value::OutputList(outputs) = call(Value::Tensor(image), vec![Value::Num(3.0)], Some(2))
    else {
        panic!("expected output list");
    };
    let counts = Tensor::try_from(&outputs[0]).unwrap();
    let locations = Tensor::try_from(&outputs[1]).unwrap();
    assert_eq!(counts.shape(), &[3, 1]);
    assert_eq!(locations.shape(), &[3, 1]);
    assert_eq!(counts.materialize_f64(), vec![1.0, 1.0, 2.0]);
    assert_eq!(locations.materialize_f64(), vec![0.0, 0.5, 1.0]);
}

#[test]
fn logical_image_uses_two_bins() {
    let logical = LogicalArray::new(vec![0, 1, 1, 0, 1, 0], vec![2, 3]).unwrap();
    let Value::Tensor(counts) = call(Value::LogicalArray(logical), vec![], None) else {
        panic!("expected counts");
    };
    assert_eq!(counts.shape(), &[2, 1]);
    assert_eq!(counts.materialize_f64(), vec![3.0, 3.0]);
}

#[test]
fn indexed_image_counts_colormap_indices() {
    let image = tensor(vec![1.0, 2.0, 3.0, 2.0], vec![2, 2], NumericDType::F64);
    let map = tensor(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![3, 3],
        NumericDType::F64,
    );
    let Value::OutputList(outputs) = call(Value::Tensor(image), vec![Value::Tensor(map)], Some(2))
    else {
        panic!("expected output list");
    };
    let counts = Tensor::try_from(&outputs[0]).unwrap();
    let locations = Tensor::try_from(&outputs[1]).unwrap();
    assert_eq!(counts.materialize_f64(), vec![1.0, 2.0, 1.0]);
    assert_eq!(locations.materialize_f64(), vec![1.0, 2.0, 3.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn colormap_argument_distinguishes_rank_zero_bin_count_from_rank_one_map() {
    let rank_zero_bins = Value::Tensor(Tensor::new(vec![3.0], Vec::new()).unwrap());
    let Value::Tensor(counts) = call(Value::Num(1.0), vec![rank_zero_bins], None) else {
        panic!("expected grayscale counts");
    };
    assert_eq!(counts.shape(), &[3, 1]);
    assert_eq!(counts.materialize_f64(), vec![0.0, 0.0, 1.0]);

    let rank_one_map = Value::Tensor(Tensor::new(vec![0.0; 3], vec![3]).unwrap());
    let error = block_on(imhist_builtin(
        None,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![rank_one_map]),
    ))
    .expect_err("rank-one map is not an Nx3 colormap");
    assert!(error.message().contains("Nx3"));

    let rank_two_map = Value::Tensor(Tensor::new(vec![0.0; 3], vec![1, 3]).unwrap());
    let Value::Tensor(counts) = call(Value::Num(1.0), vec![rank_two_map], None) else {
        panic!("expected indexed counts");
    };
    assert_eq!(counts.shape(), &[1, 1]);
    assert_eq!(counts.materialize_f64(), vec![1.0]);
}

#[test]
fn indexed_image_accepts_n_dimensional_storage() {
    let image = tensor(vec![1.0, 2.0, 3.0, 2.0], vec![1, 2, 2], NumericDType::F64);
    let map = tensor(vec![0.0; 9], vec![3, 3], NumericDType::F64);
    let Value::Tensor(counts) = call(Value::Tensor(image), vec![Value::Tensor(map)], None) else {
        panic!("expected counts");
    };
    assert_eq!(counts.materialize_f64(), vec![1.0, 2.0, 1.0]);
}

#[test]
fn uint8_indexed_image_uses_zero_based_colormap_indices() {
    let image = tensor(vec![0.0, 1.0, 1.0, 2.0], vec![2, 2], NumericDType::U8);
    let map = tensor(vec![0.0; 9], vec![3, 3], NumericDType::F64);
    let Value::Tensor(counts) = call(Value::Tensor(image), vec![Value::Tensor(map)], None) else {
        panic!("expected counts");
    };
    assert_eq!(counts.materialize_f64(), vec![1.0, 2.0, 1.0]);
}

#[test]
fn wide_integer_grayscale_input_rejects_before_lossy_conversion() {
    let image = Tensor::new_integer(
        IntegerStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX]),
        vec![1, 3],
    )
    .expect("uint64 tensor");
    let Err(err) = block_on(evaluate(Value::Tensor(image), &[])) else {
        panic!("uint64 grayscale image must be rejected");
    };
    assert!(err.message().contains("int64 and uint64"));
}

#[test]
fn wide_integer_index_errors_report_the_exact_native_value() {
    let exact = (1_u64 << 53) + 1;
    let image =
        Tensor::new_integer(IntegerStorage::U64(vec![exact]), vec![1, 1]).expect("uint64 tensor");
    let indexed = IndexedInput::from_value(Value::Tensor(image), 3).expect("indexed uint64 image");

    let Err(err) = indexed.evaluate() else {
        panic!("wide index must be rejected");
    };
    assert!(err.message().contains(&exact.to_string()));
}

#[test]
fn scalar_number_reads_typed_integer_storage_exactly() {
    let bins =
        Tensor::new_integer(IntegerStorage::U16(vec![2026]), vec![1, 1]).expect("typed bin count");

    assert_eq!(scalar_number(&Value::Tensor(bins)), Some(2026.0));
}

#[test]
fn bin_count_parser_preserves_typed_integer_scalar_bounds() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    assert_eq!(
        parse_optional_bin_count(&Value::Int(IntValue::U32(1024))).unwrap(),
        Some(1024)
    );

    let bins =
        Tensor::new_integer(IntegerStorage::U64(vec![2048]), vec![1, 1]).expect("typed bin count");
    assert_eq!(
        parse_optional_bin_count(&Value::Tensor(bins)).unwrap(),
        Some(2048)
    );

    let negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative bin count");
    assert!(parse_optional_bin_count(&Value::Tensor(negative)).is_err());
    assert!(parse_optional_bin_count(&Value::Int(IntValue::U64(MAX_BINS as u64 + 1))).is_err());

    for storage in [
        IntegerStorage::I8(vec![2]),
        IntegerStorage::I16(vec![2]),
        IntegerStorage::I32(vec![2]),
        IntegerStorage::I64(vec![2]),
        IntegerStorage::U8(vec![2]),
        IntegerStorage::U16(vec![2]),
        IntegerStorage::U32(vec![2]),
        IntegerStorage::U64(vec![2]),
    ] {
        let bins = Tensor::new_integer(storage, vec![1, 1]).expect("typed bin count");
        assert_eq!(
            parse_optional_bin_count(&Value::Tensor(bins)).unwrap(),
            Some(2)
        );
    }
}

#[test]
fn integer_capabilities_record_double_outputs_and_typed_n_extension() {
    assert_eq!(IMHIST_INTEGER_CAPABILITIES.len(), 4);
    assert!(IMHIST_INTEGER_CAPABILITIES[..3]
        .iter()
        .all(|capability| capability.output_class == BuiltinIntegerOutputClassRule::Double));
    let typed_n = &IMHIST_INTEGER_CAPABILITIES[2];
    assert_eq!(typed_n.inputs[0].name, "n");
    assert_eq!(typed_n.inputs[0].classes.len(), 8);
    assert_eq!(
        typed_n.inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
}

#[test]
fn bin_count_parser_rejects_unrepresentable_double_before_casting() {
    let boundary = if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    };
    assert!(parse_optional_bin_count(&Value::Num(boundary)).is_err());
}

#[test]
fn rejects_out_of_range_floating_grayscale_values() {
    let image = tensor(vec![0.0, 35.0, 220.0, 255.0], vec![2, 2], NumericDType::F64);
    let err = block_on(imhist_builtin(
        None,
        Value::Tensor(image),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:imhist:InvalidArgument"));
    assert!(err.message().contains("normalized to [0, 1]"));
}

#[test]
fn rejects_nan_indexed_image_values() {
    let image = tensor(vec![1.0, f64::NAN], vec![1, 2], NumericDType::F64);
    let map = tensor(vec![0.0; 6], vec![2, 3], NumericDType::F64);
    let err = block_on(imhist_builtin(
        None,
        Value::Tensor(image),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(map)]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:imhist:InvalidArgument"));
    assert!(err.message().contains("finite integer indices"));
}

#[test]
fn rejects_colormap_values_outside_unit_range() {
    let image = tensor(vec![1.0], vec![1, 1], NumericDType::F64);
    let map = tensor(vec![1.5, 0.0, 0.0], vec![1, 3], NumericDType::F64);
    let err = block_on(imhist_builtin(
        None,
        Value::Tensor(image),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(map)]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:imhist:InvalidArgument"));
    assert!(err.message().contains("range [0, 1]"));
}

#[test]
fn rejects_more_than_two_outputs() {
    let image = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
    let err = {
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        block_on(imhist_builtin(
            None,
            Value::Tensor(image),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err()
    };
    assert_eq!(err.identifier(), Some("RunMat:imhist:TooManyOutputs"));
}

#[test]
fn output_and_arity_rejection_precede_image_materialization() {
    let wide = Value::Int(IntValue::U64(u64::MAX));
    let excess = {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        block_on(imhist_builtin(
            None,
            wide.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err()
    };
    assert_eq!(excess.identifier(), Some("RunMat:imhist:TooManyOutputs"));
    let arity = block_on(imhist_builtin(
        None,
        wide,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::Num(3.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(arity.identifier(), Some("RunMat:imhist:InvalidArgument"));
}

#[test]
fn accepts_n_dimensional_grayscale_input_including_three_plane_shape() {
    let image = tensor(vec![0.0; 12], vec![2, 2, 3], NumericDType::F64);
    let Value::Tensor(counts) = block_on(imhist_builtin(
        None,
        Value::Tensor(image),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("expected counts tensor");
    };
    assert_eq!(counts.shape(), &[DEFAULT_GRAYSCALE_BINS, 1]);
    assert_eq!(counts.materialize_f64().iter().sum::<f64>(), 12.0);
    assert_eq!(counts.materialize_f64()[0], 12.0);
}

#[test]
fn resident_programmatic_outputs_are_deferred_unique_and_owner_preserving() {
    test_support::with_test_provider(|provider| {
        let input = tensor(vec![0.0, 1.0, 1.0, 255.0], vec![2, 2], NumericDType::U8);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let result = {
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            block_on(imhist_builtin(
                None,
                Value::GpuTensor(source.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("resident histogram")
        };
        let Value::OutputList(outputs) = result else {
            panic!("expected two outputs");
        };
        let [Value::GpuTensor(counts), Value::GpuTensor(locations)] = outputs.as_slice() else {
            panic!("expected resident outputs");
        };
        assert!(!same_gpu_handle(counts, &source));
        assert!(!same_gpu_handle(locations, &source));
        assert!(!same_gpu_handle(counts, locations));
        assert!(
            runmat_accelerate_api::provider_for_handle(counts).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert!(
            runmat_accelerate_api::provider_for_handle(locations).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
        );
        assert!(runmat_accelerate_api::handle_is_explicit(counts));
        assert!(runmat_accelerate_api::handle_is_explicit(locations));

        let gathered_source =
            test_support::gather(Value::GpuTensor(source)).expect("source remains readable");
        assert_eq!(gathered_source.numeric_dtype(), NumericDType::U8);
        assert_eq!(gathered_source.materialize_f64(), input.materialize_f64());
    });
}

#[test]
fn unpolled_resident_histogram_future_leaves_caller_lease_live() {
    test_support::with_test_provider(|provider| {
        let input = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let pending = imhist_builtin(
            None,
            Value::GpuTensor(source.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        );
        drop(pending);
        assert!(source.allocation_lease().validate_live().is_ok());
        let gathered = test_support::gather(Value::GpuTensor(source)).expect("caller remains live");
        assert_eq!(gathered.materialize_f64(), input.materialize_f64());
    });
}

#[test]
fn selected_host_histogram_reports_completed_owner_upload_once() {
    test_support::with_test_provider(|provider| {
        let witness = Rc::new(PresentationWitness::default());
        let runtime = witness.context();
        let input = tensor(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2], NumericDType::F64);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let result = {
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            block_on(imhist_builtin(
                Some(runtime),
                Value::GpuTensor(source.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("selected histogram")
        };
        let Value::OutputList(outputs) = result else {
            panic!("two outputs")
        };
        assert!(matches!(
            outputs.as_slice(),
            [Value::GpuTensor(_), Value::GpuTensor(_)]
        ));
        witness.drain();
        assert_eq!(witness.selections(), 1);
        assert_eq!(witness.feedback().len(), 1);
        assert!(source.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn selected_histogram_unpolled_and_statement_forms_do_not_probe() {
    test_support::with_test_provider(|provider| {
        let witness = Rc::new(PresentationWitness::default());
        let runtime = witness.context();
        let input = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::F64);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        drop(imhist_builtin(
            Some(runtime.clone()),
            Value::GpuTensor(source.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ));
        assert_eq!(witness.selections(), 0);
        let _requested = crate::output_context::push_output_count(0).expect("output-count scope");
        let counts = block_on(imhist_builtin(
            Some(runtime),
            Value::GpuTensor(source.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("statement histogram");
        assert!(matches!(counts, Value::Tensor(_)));
        assert_eq!(witness.selections(), 0);
        assert!(witness.feedback().is_empty());
        assert!(source.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn readback_workspace_overflow_fails_admission_before_provider_effects() {
    test_support::with_test_provider(|provider| {
        let input = tensor(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2], NumericDType::F64);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let request = || crate::context::host_presentation::ExactHostUpload {
            operation: NAME,
            variant: "overflow.admission.test",
            owner: &source,
            domain_facts: runmat_execution::Digest::sha256(b"overflow-admission"),
            outputs: vec![runmat_accelerate_api::ProviderRepresentation {
                element_type: runmat_accelerate_api::ProviderElementType::F64,
                storage: runmat_accelerate_api::ProviderStorage::DenseReal,
                layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
                shape: vec![3, 1],
                residency: runmat_accelerate_api::ProviderResidency::Device,
            }],
            host_compute_scratch: runmat_execution::ResourceByteBound::Known(u64::MAX),
        };
        provider.reset_telemetry();
        assert!(
            crate::context::host_presentation::select_exact_host_upload(None, request())
                .unwrap()
                .is_none()
        );
        let witness = Rc::new(PresentationWitness::default());
        let runtime = witness.context();
        let error =
            crate::context::host_presentation::select_exact_host_upload(Some(&runtime), request())
                .err()
                .expect("overflow must fail before policy selection");
        assert_eq!(witness.selections(), 0);
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:ResourceOverflow")
        );
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.download_bytes, 0);
        assert_eq!(telemetry.upload_bytes, 0);
        assert!(source.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn resident_statement_form_returns_host_counts_without_plotting_or_publication() {
    test_support::with_test_provider(|provider| {
        let input = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::F64);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let result = {
            let _requested =
                crate::output_context::push_output_count(0).expect("output-count scope");
            block_on(imhist_builtin(
                None,
                Value::GpuTensor(source.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("resident statement histogram")
        };
        let Value::Tensor(counts) = result else {
            panic!("gpuArray statement form must return host counts in ans");
        };
        assert_eq!(counts.shape(), &[DEFAULT_GRAYSCALE_BINS, 1]);
        let gathered_source = test_support::gather(Value::GpuTensor(source))
            .expect("source remains resident and readable");
        assert_eq!(gathered_source.materialize_f64(), input.materialize_f64());
    });
}

#[test]
fn explicit_resident_input_preserves_unsupported_typed_upload_error() {
    test_support::with_f32_test_provider(|provider| {
        runmat_accelerate::ensure_residency_hooks();
        let input = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::F32);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let err = block_on(imhist_builtin(
            None,
            Value::GpuTensor(source),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("double output cannot silently become host output");
        assert!(
            err.message().contains("failed to restore GPU result")
                && err
                    .message()
                    .contains("cannot store native double payloads"),
            "unexpected error: {}",
            err.message()
        );
    });
}

#[test]
fn automatic_f32_resident_outputs_fall_back_to_host_without_publication() {
    test_support::with_f32_test_provider(|provider| {
        runmat_accelerate::ensure_residency_hooks();
        let input = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::F32);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("publish provenance view");
        runmat_accelerate_api::mark_residency(&source);
        let result = {
            let _outputs =
                crate::output_count::push_output_count(Some(2)).expect("output-count scope");
            block_on(imhist_builtin(
                None,
                Value::GpuTensor(source.clone()),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .expect("automatic precision mismatch falls back")
        };
        let Value::OutputList(outputs) = result else {
            panic!("expected two outputs");
        };
        assert!(matches!(
            outputs.as_slice(),
            [Value::Tensor(_), Value::Tensor(_)]
        ));
        assert!(runmat_accelerate::fusion_residency::is_resident(&source));
    });
}

#[test]
fn indexed_resident_image_rejects_before_download() {
    test_support::with_test_provider(|provider| {
        let input = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
        let source = gpu_helpers::upload_tensor(provider, &input).expect("resident image");
        let source = source
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("publish provenance view");
        let map = tensor(vec![0.0; 6], vec![2, 3], NumericDType::F64);
        let err = block_on(imhist_builtin(
            None,
            Value::GpuTensor(source.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(map)]),
        ))
        .expect_err("indexed gpuArray input must reject");
        assert_eq!(err.identifier(), Some("RunMat:imhist:UnsupportedImage"));
        assert!(err.message().contains("indexed gpuArray"));
        let gathered = test_support::gather(Value::GpuTensor(source))
            .expect("rejected source remains readable");
        assert_eq!(gathered.materialize_f64(), input.materialize_f64());
    });
}

#[cfg(feature = "plot-core")]
#[test]
fn plotting_downsamples_large_histograms_without_changing_outputs() {
    let image = tensor(vec![0.0, 65535.0], vec![1, 2], NumericDType::U16);
    let eval = block_on(evaluate(
        Value::Tensor(image),
        &[Value::Num((MAX_PLOT_BINS + 100) as f64)],
    ))
    .unwrap();
    assert_eq!(eval.counts.materialize_f64().len(), MAX_PLOT_BINS + 100);
    let plot = plot_display_bins(&eval.counts, &eval.locations).unwrap();
    assert!(plot.counts.len() <= MAX_PLOT_BINS);
    assert_eq!(plot.counts.iter().sum::<f64>(), 2.0);
}

#[cfg(feature = "plot-core")]
#[test]
fn statement_form_renders_bar_chart_without_value() {
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{
        clear_figure, clone_figure, current_figure_handle, reset_hold_state_for_run,
    };
    use runmat_plot::plots::PlotElement;

    let _guard = lock_plot_registry();
    ensure_plot_test_env();
    reset_hold_state_for_run();
    let _ = clear_figure(None);
    let _requested = crate::output_context::push_output_count(0).expect("output-count scope");
    let image = tensor(vec![0.0, 1.0, 1.0, 2.0], vec![2, 2], NumericDType::U8);
    let out = block_on(imhist_builtin(
        None,
        Value::Tensor(image),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    assert_eq!(out, Value::output_list(Vec::new()));
    if let Some(fig) = clone_figure(current_figure_handle()) {
        if let Some(plot) = fig.plots().next() {
            assert!(matches!(plot, PlotElement::Bar(_)));
        }
    }
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn statement_form_noops_without_plot_core() {
    let _requested = crate::output_context::push_output_count(0).expect("output-count scope");
    let image = tensor(vec![0.0, 1.0], vec![1, 2], NumericDType::U8);
    let out = block_on(imhist_builtin(
        None,
        Value::Tensor(image),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    assert_eq!(out, Value::output_list(Vec::new()));
}
