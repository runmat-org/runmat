use super::*;
use crate::builtins::common::random;
use runmat_value::{IntegerStorage, StructValue};

#[test]
fn option_spellings_and_preflight_are_typed_and_preeffect() {
    assert_eq!(
        TsneOption::from_keyword("NumDimensions"),
        Some(TsneOption::NumDimensions)
    );
    assert_eq!(
        DistanceMetric::from_keyword("fastseuclidean"),
        Some(DistanceMetric::Seuclidean)
    );
    assert_eq!(Algorithm::from_keyword("exact"), Some(Algorithm::Exact));
    assert!(preflight_option_headers(&[Value::from("Distance")]).is_err());
    assert!(preflight_option_headers(&[Value::from("unknown"), Value::from("value")]).is_err());
    assert!(preflight_observations(&Value::from("not observations")).is_err());
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn reset_rng() -> impl Drop {
    let guard = random::test_guard();
    random::reset_rng();
    guard
}

#[tokio::test]
async fn tsne_embeds_rows_and_returns_loss() {
    let _guard = reset_rng();
    let _out_guard = crate::output_count::push_output_count(Some(2));
    let value = tsne_builtin(
        tensor(vec![0.0, 0.1, 5.0, 5.1, 0.0, 0.1, 5.0, 5.1], vec![4, 2]),
        vec![
            Value::String("Algorithm".into()),
            Value::String("exact".into()),
            Value::String("Perplexity".into()),
            Value::Num(2.0),
            Value::String("Options".into()),
            {
                let mut options = StructValue::new();
                options.insert("MaxIter", Value::Num(50.0));
                Value::Struct(options)
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::OutputList(outputs) => {
            match &outputs[0] {
                Value::Tensor(y) => {
                    assert_eq!(y.shape, vec![4, 2]);
                    assert!(y.materialize_f64().iter().all(|value| value.is_finite()));
                }
                other => panic!("expected tensor, got {other:?}"),
            }
            assert!(matches!(&outputs[1], Value::Num(loss) if loss.is_finite()));
        }
        other => panic!("expected outputs, got {other:?}"),
    }
}

#[tokio::test]
async fn tsne_removes_nan_rows_and_honors_num_dimensions() {
    let _guard = reset_rng();
    let value = tsne_builtin(
        tensor(
            vec![
                0.0,
                1.0,
                f64::NAN,
                3.0, //
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            vec![4, 2],
        ),
        vec![
            Value::String("NumDimensions".into()),
            Value::Num(3.0),
            Value::String("InitialY".into()),
            Value::Tensor(Tensor::new(vec![0.0; 12], vec![4, 3]).unwrap()),
            Value::String("Options".into()),
            {
                let mut options = StructValue::new();
                options.insert("MaxIter", Value::Num(2.0));
                Value::Struct(options)
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::Tensor(y) => assert_eq!(y.shape, vec![3, 3]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[tokio::test]
async fn tsne_supports_standardize_pca_and_cosine_distance() {
    let _guard = reset_rng();
    let value = tsne_builtin(
        tensor(
            vec![0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 4.0, 3.0, 2.0, 1.0],
            vec![4, 3],
        ),
        vec![
            Value::String("Standardize".into()),
            Value::Bool(true),
            Value::String("NumPCAComponents".into()),
            Value::Num(2.0),
            Value::String("Distance".into()),
            Value::String("cosine".into()),
            Value::String("Options".into()),
            {
                let mut options = StructValue::new();
                options.insert("MaxIter", Value::Num(5.0));
                Value::Struct(options)
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::Tensor(y) => assert_eq!(y.shape, vec![4, 2]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[tokio::test]
async fn tsne_reads_typed_integer_data_initial_y_and_options_exactly() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = reset_rng();
    let value = tsne_builtin(
        poisoned_int_tensor(
            IntegerStorage::I16(vec![0, 1, 5, 6, 0, 1, 5, 6]),
            vec![4, 2],
        ),
        vec![
            Value::String("NumDimensions".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            Value::String("NumPCAComponents".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            Value::String("Perplexity".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![2]), vec![1, 1]),
            Value::String("Standardize".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            Value::String("InitialY".into()),
            poisoned_int_tensor(IntegerStorage::I16(vec![0; 8]), vec![4, 2]),
            Value::String("Options".into()),
            {
                let mut options = StructValue::new();
                options.insert(
                    "MaxIter",
                    cleared_int_tensor(IntegerStorage::U16(vec![2]), vec![1, 1]),
                );
                Value::Struct(options)
            },
        ],
    )
    .await
    .unwrap();
    match value {
        Value::Tensor(y) => assert_eq!(y.shape, vec![4, 2]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn tsne_gates_integer_roles_and_explicit_residency_before_gather() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = poisoned_int_tensor(IntegerStorage::U8(vec![1]), vec![1, 1]);
    let error = ensure_tsne_extensions(&integer, &[]).expect_err("integer data must gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:TsneIntegerDataExtension")
    );
    let error = ensure_tsne_extensions(&Value::Num(1.0), &[Value::from("NumDimensions"), integer])
        .expect_err("integer option must gate");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:TsneIntegerOptionExtension")
    );
    let explicit = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let explicit = explicit
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let error = ensure_tsne_extensions(&Value::GpuTensor(explicit), &[])
        .expect_err("explicit residency must gate before provider access");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:TsneResidentInputExtension")
    );
}

#[test]
fn tsne_rejects_lossy_integer_floating_boundaries() {
    let wide = poisoned_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1]);
    let error = ensure_exact_tsne_integer_value(&wide, "observation data")
        .expect_err("lossy integer must reject");
    assert!(error.message().contains("exactly representable"));
}

#[tokio::test]
async fn tsne_rejects_bad_initial_y_shape() {
    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![
            Value::String("InitialY".into()),
            Value::Tensor(Tensor::new(vec![0.0; 3], vec![3, 1]).unwrap()),
        ],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("InitialY"));
}

#[test]
fn tsne_empty_option_helpers_use_typed_integer_storage_len() {
    let empty = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();
    let empty = Value::Tensor(empty);

    assert!(is_empty_numeric(&empty));
    assert!(is_empty_option_value(&empty));
}

#[tokio::test]
async fn tsne_rejects_unbounded_resource_options() {
    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![Value::String("NumDimensions".into()), Value::Num(1.0e20)],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("NumDimensions"));

    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![Value::String("Options".into()), {
            let mut options = StructValue::new();
            options.insert("MaxIter", Value::Num(1.0e20));
            Value::Struct(options)
        }],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("MaxIter"));
}

#[tokio::test]
async fn tsne_rejects_dense_feature_work_before_allocation() {
    let x = Tensor::new(vec![1.0; 2 * 2001], vec![2, 2001]).unwrap();
    let err = tsne_builtin(
        Value::Tensor(x),
        vec![
            Value::String("Distance".into()),
            Value::String("mahalanobis".into()),
        ],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("dense feature cells"));
}

#[tokio::test]
async fn tsne_rejects_inf_even_after_nan_in_row() {
    let err = tsne_builtin(
        tensor(vec![f64::NAN, 1.0, f64::INFINITY, 2.0], vec![2, 2]),
        Vec::new(),
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("Inf"));
}

#[tokio::test]
async fn tsne_rejects_unsupported_fast_distance_names() {
    let err = tsne_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        vec![
            Value::String("Distance".into()),
            Value::String("fastcityblock".into()),
        ],
    )
    .await
    .unwrap_err();
    assert!(err.message.contains("unsupported Distance"));
}
