use futures::executor::block_on;
use runmat_value::{IntegerStorage, Tensor};

use super::admission::is_empty_option_value;
use super::*;
use runmat_builtins::{KmeansDistance, KmeansEmptyAction};

#[test]
fn registered_kmeans_observes_its_invocation_cancellation_before_materialization() {
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;

    let source = CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    source.cancel(CancellationReason::User);
    let error = block_on(kmeans_builtin_registered(
        Some(runtime),
        tensor(vec![1.0, 2.0], 2, 1),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect_err("cancelled invocation must not prepare observations");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn preparation_yields_on_real_observation_work_and_preserves_native_values() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let expected = prepare_data(
        tensor(vec![1.0, 3.0, 2.0, 4.0], 2, 2),
        Distance::SqEuclidean,
    )
    .expect("native preparation");
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let actual = block_on(geometry::prepare_data_cooperative(
        tensor(vec![1.0, 3.0, 2.0, 4.0], 2, 2),
        Distance::SqEuclidean,
        &mut cooperation::KmeansCooperation::with_budget(budget),
    ))
    .expect("cooperative preparation");
    assert_eq!(actual.rows, expected.rows);
    assert_eq!(actual.original_to_valid, expected.original_to_valid);
    assert!(turns.get() >= 4);
}

#[test]
fn clustering_numerics_match_with_and_without_forced_cpu_turns() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let data = prepare_data(
        tensor(vec![0.0, 1.0, 9.0, 10.0], 4, 1),
        Distance::SqEuclidean,
    )
    .unwrap();
    let options = Options {
        start: StartSpec::Numeric(vec![vec![0.0], vec![10.0]]),
        ..Options::default()
    };
    let expected = block_on(compute::run_kmeans_cooperative(
        &data,
        2,
        &options,
        &mut cooperation::KmeansCooperation::new(None),
    ))
    .unwrap();
    let source = CancellationSource::new();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let actual = block_on(compute::run_kmeans_cooperative(
        &data,
        2,
        &options,
        &mut cooperation::KmeansCooperation::with_budget(budget),
    ))
    .unwrap();
    assert_eq!(actual.idx_valid, expected.idx_valid);
    assert_eq!(actual.centers, expected.centers);
    assert_eq!(actual.sumd, expected.sumd);
    assert_eq!(actual.distances, expected.distances);
    assert_eq!(actual.objective.to_bits(), expected.objective.to_bits());
    assert!(turns.get() > 20);
}

#[test]
fn cancellation_during_assignment_prevents_a_partial_replicate() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let data = prepare_data(
        tensor(vec![0.0, 1.0, 9.0, 10.0], 4, 1),
        Distance::SqEuclidean,
    )
    .unwrap();
    let options = Options {
        start: StartSpec::Numeric(vec![vec![0.0], vec![10.0]]),
        ..Options::default()
    };
    let source = Rc::new(CancellationSource::new());
    let turns = Rc::new(Cell::new(0));
    let cancel_on_turn = source.clone();
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == 12 {
                cancel_on_turn.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let error = block_on(compute::run_kmeans_cooperative(
        &data,
        2,
        &options,
        &mut cooperation::KmeansCooperation::with_budget(budget),
    ))
    .expect_err("cancelled assignment must not publish a replicate");
    assert_eq!(turns.get(), 12);
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
}

#[test]
fn cancellation_during_output_materialization_prevents_publication() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let data = prepare_data(
        tensor(vec![0.0, 1.0, 9.0, 10.0], 4, 1),
        Distance::SqEuclidean,
    )
    .unwrap();
    let result = ReplicateResult {
        idx_valid: vec![1, 1, 2, 2],
        centers: vec![vec![0.5], vec![9.5]],
        sumd: vec![0.5, 0.5],
        distances: vec![vec![0.25, 90.25]; 4],
        objective: 1.0,
    };
    let source = Rc::new(CancellationSource::new());
    let turns = Rc::new(Cell::new(0));
    let cancel_on_turn = source.clone();
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == 3 {
                cancel_on_turn.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let error = block_on(output::outputs_for_count_cooperative(
        result,
        &data,
        &mut cooperation::KmeansCooperation::with_budget(budget),
    ))
    .expect_err("cancelled output materialization cannot publish a partial result");
    assert_eq!(turns.get(), 3);
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(matches!(
        error
            .source
            .as_deref()
            .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>()),
        Some(crate::execution::ExecutionServiceError::Cancelled)
    ));
}

#[test]
fn native_integer_exactness_scan_is_cooperative_without_a_f64_copy() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use runmat_value::IntegerStorage;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    let source = Rc::new(CancellationSource::new());
    let turns = Rc::new(Cell::new(0));
    let cancel_on_turn = source.clone();
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == 4 {
                cancel_on_turn.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let value = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 54; 16]), vec![16, 1]).unwrap(),
    );
    let error = block_on(admission::numeric_value_is_exact_cooperative(
        &value,
        &mut cooperation::KmeansCooperation::with_budget(budget),
    ))
    .expect_err("cancelled integer scan must not silently pass admission");
    assert_eq!(turns.get(), 4);
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(block_on(admission::numeric_value_is_exact_cooperative(
        &value,
        &mut cooperation::KmeansCooperation::new(None),
    ))
    .unwrap());
}
fn tensor(data: Vec<f64>, rows: usize, cols: usize) -> Value {
    Value::Tensor(Tensor::new(data, vec![rows, cols]).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn poisoned_int_tensor_shape(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

#[test]
fn public_option_spellings_decode_once_into_typed_controls() {
    assert_eq!(
        KmeansOption::from_keyword("DISTANCE"),
        Some(KmeansOption::Distance)
    );
    assert_eq!(
        KmeansDistance::from_keyword("city block"),
        Some(KmeansDistance::Cityblock)
    );
    assert_eq!(
        KmeansEmptyAction::from_keyword("drop"),
        Some(KmeansEmptyAction::Drop)
    );
    assert_eq!(
        KmeansStart::from_keyword("cluster"),
        Some(KmeansStart::Cluster)
    );
    assert!(KmeansOption::from_keyword("unrecognized").is_none());
}

#[test]
fn cosine_and_correlation_prepare_large_and_small_finite_rows() {
    let x = tensor(vec![1.0e200, 1.0e-200, 2.0e200, 2.0e-200], 2, 2);
    for metric in [Distance::Cosine, Distance::Correlation] {
        let prepared = prepare_data(x.clone(), metric).expect("finite observations");
        assert_eq!(prepared.rows.len(), 2);
        for row in &prepared.rows {
            assert!(row.iter().all(|value| value.is_finite()));
            assert!((row.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() < 1.0e-13);
        }
        assert!((prepared.rows[0][0] - prepared.rows[1][0]).abs() < 1.0e-13);
        assert!((prepared.rows[0][1] - prepared.rows[1][1]).abs() < 1.0e-13);
    }
}

#[test]
fn malformed_public_headers_and_geometry_fail_before_solver() {
    assert!(parse_option_headers(&[Value::from("Distance")]).is_err());
    assert!(parse_option_headers(&[Value::from("unknown"), Value::from("value")]).is_err());
    assert!(preflight_observations(&Value::from("not observations")).is_err());
    assert!(preflight_cluster_count(&tensor(vec![1.0, 2.0], 1, 2)).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn kmeans_preserves_rank_zero_and_rank_one_observation_geometry() {
    let start = || Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());

    let rank_zero = Value::Tensor(Tensor::new(vec![1.0], Vec::new()).unwrap());
    let Value::Tensor(indices) = block_on(kmeans_builtin(
        rank_zero,
        Value::Num(1.0),
        vec![Value::from("Start"), start()],
    ))
    .expect("rank-zero input is one observation with one feature") else {
        panic!("expected index tensor");
    };
    assert_eq!(indices.shape(), [1, 1]);
    assert_eq!(indices.materialize_f64(), vec![1.0]);

    let rank_one = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
    let Value::Tensor(indices) = block_on(kmeans_builtin(
        rank_one,
        Value::Num(1.0),
        vec![Value::from("Start"), start()],
    ))
    .expect("rank-one input is three observations with one feature") else {
        panic!("expected index tensor");
    };
    assert_eq!(indices.shape(), [3, 1]);
    assert_eq!(indices.materialize_f64(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn option_parsers_read_all_integer_storage_variants() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];
    for storage in storages {
        let value = poisoned_int_tensor(storage, 1, 1);
        assert_eq!(parse_positive_usize(&value, "k").unwrap(), 1);
        assert!(bool_option(&value).unwrap());
    }
}

fn cleared_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn outputs(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique output list")
            .try_into_unbounded_vec()
            .expect("unbounded output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

#[test]
fn kmeans_documented_floating_gpu_fallback_preserves_explicit_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        random::reset_rng();
        let input = Tensor::new(vec![0.0, 1.0, 9.0, 10.0], vec![4, 1]).unwrap();
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
            .expect("upload observations");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance");
        let output = block_on(kmeans_builtin(
            Value::GpuTensor(handle),
            Value::Num(2.0),
            vec![
                Value::from("Start"),
                Value::Tensor(Tensor::new(vec![0.0, 10.0], vec![2, 1]).unwrap()),
            ],
        ))
        .expect("documented resident kmeans");
        assert!(matches!(output, Value::GpuTensor(_)));
    });
}

#[test]
#[cfg(feature = "wgpu")]
fn kmeans_wgpu_fallback_enforces_explicit_residency_contract() {
    let _accel_guard = crate::builtins::common::test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU provider");
    random::reset_rng();
    let input = Tensor::new(vec![0.0, 1.0, 9.0, 10.0], vec![4, 1]).unwrap();
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &input)
        .expect("upload observations");
    let handle = handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let result = block_on(kmeans_builtin(
        Value::GpuTensor(handle.clone()),
        Value::Num(2.0),
        vec![
            Value::from("Start"),
            Value::Tensor(Tensor::new(vec![0.0, 10.0], vec![2, 1]).unwrap()),
        ],
    ));
    let Value::GpuTensor(output) = result.expect("documented WGPU kmeans") else {
        panic!("expected resident output");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(
        output.device_id(),
        runmat_accelerate_api::AccelProvider::device_id(provider)
    );
    assert_eq!(output.shape(), &[4, 1]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&output),
        Some(runmat_accelerate_api::ProviderPrecision::F64)
    );
}

#[test]
fn kmeans_clusters_numeric_matrix_with_outputs() {
    let _rng_guard = random::test_guard();
    random::reset_rng();
    let x = tensor(vec![0.0, 0.2, 9.8, 10.0, 0.0, 0.1, 9.9, 10.1], 4, 2);
    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let out = block_on(kmeans_builtin(
        x,
        Value::Num(2.0),
        vec![
            Value::from("Start"),
            tensor(vec![0.0, 10.0, 0.0, 10.0], 2, 2),
            Value::from("MaxIter"),
            Value::Num(20.0),
        ],
    ))
    .unwrap();
    let out = outputs(out);
    match &out[0] {
        Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 2.0, 2.0]),
        other => panic!("idx {other:?}"),
    }
    match &out[1] {
        Value::Tensor(c) => {
            assert_eq!(c.shape(), [2, 2]);
            assert!((c.materialize_f64()[0] - 0.1).abs() < 1.0e-12);
            assert!((c.materialize_f64()[1] - 9.9).abs() < 1.0e-12);
            assert!((c.materialize_f64()[2] - 0.05).abs() < 1.0e-12);
            assert!((c.materialize_f64()[3] - 10.0).abs() < 1.0e-12);
        }
        other => panic!("centers {other:?}"),
    }
    match &out[2] {
        Value::Tensor(sumd) => {
            assert_eq!(sumd.shape(), [2, 1]);
            assert!(sumd.materialize_f64().iter().all(|v| v.is_finite()));
        }
        other => panic!("sumd {other:?}"),
    }
    match &out[3] {
        Value::Tensor(d) => assert_eq!(d.shape(), [4, 2]),
        other => panic!("distances {other:?}"),
    }
}

#[test]
fn kmeans_reads_typed_integer_inputs_start_and_options_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let mut statset = runmat_value::UnboundedStructValueBuilder::new();
    statset.insert(
        "MaxIter",
        cleared_int_tensor(IntegerStorage::U16(vec![20]), 1, 1),
    );
    statset.insert(
        "TolFun",
        cleared_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
    );
    statset.insert(
        "UseParallel",
        cleared_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
    );

    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(kmeans_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 0, 10, 10]), 4, 1),
        poisoned_int_tensor(IntegerStorage::U8(vec![2]), 1, 1),
        vec![
            Value::from("Start"),
            poisoned_int_tensor(IntegerStorage::I16(vec![0, 10]), 2, 1),
            Value::from("Replicates"),
            poisoned_int_tensor(IntegerStorage::U8(vec![1]), 1, 1),
            Value::from("Options"),
            Value::Struct(statset.finish()),
        ],
    ))
    .unwrap();
    let out = outputs(out);
    match &out[0] {
        Value::Tensor(idx) => assert_eq!(idx.materialize_f64(), vec![1.0, 1.0, 2.0, 2.0]),
        other => panic!("idx {other:?}"),
    }
    match &out[1] {
        Value::Tensor(centers) => assert_eq!(centers.materialize_f64(), vec![0.0, 10.0]),
        other => panic!("centers {other:?}"),
    }
}

#[test]
fn kmeans_empty_option_helpers_use_typed_integer_storage_len() {
    let empty = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();
    let empty = Value::Tensor(empty);

    assert!(is_empty_numeric(&empty));
    assert!(is_empty_option_value(&empty));
}

#[test]
fn kmeans_treats_vector_as_column_and_nan_rows_as_missing() {
    let _rng_guard = random::test_guard();
    random::reset_rng();
    let x = tensor(vec![1.0, 2.0, f64::NAN, 20.0, 21.0], 1, 5);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(kmeans_builtin(
        x,
        Value::Num(2.0),
        vec![Value::from("Start"), Value::from("sample")],
    ))
    .unwrap();
    let out = outputs(out);
    match &out[0] {
        Value::Tensor(idx) => {
            assert_eq!(idx.shape(), [5, 1]);
            assert!(idx.materialize_f64()[2].is_nan());
            assert!(idx.materialize_f64()[0].is_finite());
            assert!(idx.materialize_f64()[4].is_finite());
        }
        other => panic!("idx {other:?}"),
    }
}

#[test]
fn kmeans_supports_cityblock_and_replicates() {
    let _rng_guard = random::test_guard();
    random::reset_rng();
    let x = tensor(vec![0.0, 1.0, 10.0, 11.0], 4, 1);
    let out = block_on(kmeans_builtin(
        x,
        Value::Num(2.0),
        vec![
            Value::from("Distance"),
            Value::from("cityblock"),
            Value::from("Replicates"),
            Value::Num(2.0),
            Value::from("Start"),
            Value::from("plus"),
        ],
    ))
    .unwrap();
    match out {
        Value::Tensor(idx) => assert_eq!(idx.shape(), [4, 1]),
        other => panic!("idx {other:?}"),
    }
}

#[test]
fn kmeans_infers_k_from_numeric_start_when_k_is_empty() {
    let x = tensor(vec![0.0, 0.5, 10.0, 10.5], 4, 1);
    let empty_k = Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap());
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(kmeans_builtin(
        x,
        empty_k,
        vec![
            Value::from("Start"),
            tensor(vec![0.0, 10.0], 2, 1),
            Value::from("MaxIter"),
            Value::Num(5.0),
        ],
    ))
    .unwrap();
    let out = outputs(out);
    match &out[1] {
        Value::Tensor(c) => assert_eq!(c.shape(), [2, 1]),
        other => panic!("centers {other:?}"),
    }
}

#[test]
fn kmeans_accepts_3d_numeric_starts_as_replicates() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = tensor(vec![0.0, 0.2, 10.0, 10.2], 4, 1);
    let starts = poisoned_int_tensor_shape(IntegerStorage::I16(vec![0, 10, 0, 10]), vec![2, 1, 2]);
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let out = block_on(kmeans_builtin(
        x,
        Value::Num(2.0),
        vec![Value::from("Start"), starts],
    ))
    .unwrap();
    let out = outputs(out);
    match &out[2] {
        Value::Tensor(sumd) => assert_eq!(sumd.shape(), [2, 1]),
        other => panic!("sumd {other:?}"),
    }
}

#[test]
fn kmeans_integer_roles_are_independently_gated_and_declared() {
    let x = poisoned_int_tensor(IntegerStorage::I16(vec![0, 10]), 2, 1);
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(kmeans_builtin(x, Value::Num(1.0), Vec::new()))
        .expect_err("strict mode rejects typed observation data");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:KmeansIntegerObservationDataExtension")
    );
    assert_eq!(KMEANS_EXTENSIONS.len(), 5);
    assert_eq!(KMEANS_INTEGER_CAPABILITIES.len(), 5);
}

#[test]
fn kmeans_runmat_integer_data_still_requires_exact_double_values() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let x = poisoned_int_tensor(IntegerStorage::U64(vec![9_007_199_254_740_993]), 1, 1);
    let error = block_on(kmeans_builtin(x, Value::Num(1.0), Vec::new()))
        .expect_err("wide integer data cannot cross the floating boundary");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn kmeans_supports_cluster_start_and_online_phase() {
    let _rng_guard = random::test_guard();
    random::reset_rng();
    let x = tensor(vec![0.0, 0.1, 9.9, 10.0, 0.0, 0.2, 10.1, 10.0], 4, 2);
    let out = block_on(kmeans_builtin(
        x,
        Value::Num(2.0),
        vec![
            Value::from("Start"),
            Value::from("cluster"),
            Value::from("OnlinePhase"),
            Value::from("on"),
        ],
    ))
    .unwrap();
    match out {
        Value::Tensor(idx) => assert_eq!(idx.shape(), [4, 1]),
        other => panic!("idx {other:?}"),
    }
}

#[test]
fn kmeans_rejects_bad_options() {
    let x = tensor(vec![0.0, 1.0, 2.0], 3, 1);
    let err = block_on(kmeans_builtin(
        x.clone(),
        Value::Num(2.0),
        vec![Value::from("Distance"), Value::from("mahalanobis")],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:kmeans:InvalidArgument"));

    let err = block_on(kmeans_builtin(
        x,
        Value::Num(2.0),
        vec![
            Value::from("Distance"),
            Value::from("hamming"),
            Value::from("Start"),
            Value::from("uniform"),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:kmeans:InvalidArgument"));
}
