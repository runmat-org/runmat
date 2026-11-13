use super::*;
async fn gridded_interpolant_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<crate::execution::RootedValue> {
    super::gridded_interpolant_builtin(
        runtime,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}
fn parse_constructor(args: Vec<Value>) -> BuiltinResult<InterpolantSpec> {
    super::parse_constructor(
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
}
use crate::{OBJECT_INDEX_MEMBER, OBJECT_INDEX_PAREN};
use futures::executor::block_on;
use runmat_value::{IntegerStorage, NumericStorage};
use std::rc::Rc;

struct ManagedSplineService(crate::context::host_effect::ManagedHostEffectLedger);

impl crate::context::RuntimeParallelService for ManagedSplineService {
    fn supports(&self, _: crate::context::ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> crate::context::RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        crate::context::RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..crate::context::RuntimeParallelResources::default()
        }
    }
    fn managed_host_effect_ledger(
        &self,
    ) -> Option<crate::context::host_effect::ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

async fn gridded_interpolant_subsref(
    obj: Value,
    kind: String,
    payload: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let subscript = crate::object::indexing::standard_substruct_fixture_from_parts(&kind, payload)?;
    super::gridded_interpolant_subsref(obj, subscript).await
}

fn row(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).unwrap())
}

fn col(values: &[f64]) -> Value {
    Value::Tensor(Tensor::new(values.to_vec(), vec![values.len(), 1]).unwrap())
}

fn int_col(storage: IntegerStorage) -> Value {
    let len = storage.len();
    let tensor = Tensor::new_integer(storage, vec![len, 1]).unwrap();
    Value::Tensor(tensor)
}

fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).unwrap();
    Value::Tensor(tensor)
}

fn call(args: Vec<Value>) -> BuiltinResult<Value> {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    block_on(gridded_interpolant_builtin(None, args))
        // Test-only adapter: interpolant fields are numeric/text owners, not
        // newly published GC children. Clone while the original roots live.
        .map(|rooted| rooted.value().clone())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn constructor_preserves_rank_one_and_full_rank_grid_geometry() {
    let vector = Tensor::new(vec![10.0, 20.0, 40.0], vec![3]).expect("rank-one values");
    let vector_spec = parse_constructor(vec![Value::Tensor(vector)]).expect("vector constructor");
    assert_eq!(vector_spec.values.shape(), [3]);
    assert_eq!(vector_spec.grid_vectors, vec![vec![1.0, 2.0, 3.0]]);

    let volume = Tensor::new((0..8).map(f64::from).collect(), vec![2, 2, 2])
        .expect("three-dimensional values");
    let volume_spec = parse_constructor(vec![Value::Tensor(volume)]).expect("volume constructor");
    assert_eq!(volume_spec.values.shape(), [2, 2, 2]);
    assert_eq!(
        volume_spec.grid_vectors,
        vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]]
    );

    let scalar = Tensor::new(vec![7.0], Vec::new()).expect("rank-zero scalar");
    let error = match parse_constructor(vec![Value::Tensor(scalar)]) {
        Ok(_) => panic!("a scalar cannot provide the required two-point grid"),
        Err(error) => error,
    };
    assert!(error.message().contains("at least two points"));
}

fn subsref(obj: Value, args: Vec<Value>) -> BuiltinResult<crate::execution::RootedValue> {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
    let len = args.len();
    let payload = Value::Cell(CellArray::new_with_shape(args, vec![1, len]).unwrap());
    block_on(gridded_interpolant_subsref(
        obj,
        OBJECT_INDEX_PAREN.to_string(),
        payload,
    ))
}

#[test]
fn gridded_integer_extensions_and_capabilities_are_declared() {
    assert_eq!(GRIDDED_INTERPOLANT_EXTENSIONS.len(), 3);
    assert_eq!(GRIDDED_INTERPOLANT_INTEGER_CAPABILITIES.len(), 3);
}

#[test]
fn gridded_strict_mode_rejects_integer_values() {
    let _guard = crate::compatibility::push_runmat_extensions_enabled(false);
    let values = int_col(IntegerStorage::I16(vec![1, 4, 9]));
    let error = block_on(gridded_interpolant_builtin(None, vec![values]))
        .expect_err("integer values are an extension");
    assert_eq!(
        error.identifier(),
        GRIDDED_INTEGER_VALUES_EXTENSION.error_identifier
    );
}

#[test]
fn gridded_rejects_inexact_integer_grid_values_and_queries() {
    let inexact = 9_007_199_254_740_993_u64;
    let error = call(vec![
        int_col(IntegerStorage::U64(vec![inexact, inexact + 2])),
        col(&[1.0, 2.0]),
    ])
    .expect_err("inexact grid must reject");
    assert!(error.message().contains("exactly representable as double"));

    let error = call(vec![int_col(IntegerStorage::U64(vec![
        inexact,
        inexact + 2,
    ]))])
    .expect_err("inexact values must reject");
    assert!(error.message().contains("exactly representable as double"));

    let obj = call(vec![col(&[0.0, 1.0]), col(&[0.0, 1.0])]).expect("interpolant");
    let error = subsref(obj, vec![int_col(IntegerStorage::U64(vec![inexact]))])
        .expect_err("inexact query must reject");
    assert!(error.message().contains("exactly representable as double"));
}

#[test]
fn gridded_integer_query_syntaxes_are_consistent() {
    let one_d = call(vec![col(&[0.0, 1.0, 2.0]), col(&[0.0, 10.0, 20.0])])
        .expect("one-dimensional interpolant");
    match subsref(one_d, vec![int_col(IntegerStorage::I16(vec![0, 2]))])
        .expect("integer vector query")
        .value()
    {
        Value::Tensor(output) => assert_eq!(output.materialize_f64(), vec![0.0, 20.0]),
        other => panic!("expected tensor, got {other:?}"),
    }

    let grid = CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2])
        .expect("grid");
    let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).expect("values");
    let two_d =
        call(vec![Value::Cell(grid), Value::Tensor(values)]).expect("two-dimensional interpolant");
    match subsref(
        two_d,
        vec![
            int_col(IntegerStorage::I16(vec![0, 1])),
            int_col(IntegerStorage::U16(vec![0, 2])),
        ],
    )
    .expect("per-dimension integer query")
    .value()
    {
        Value::Tensor(output) => assert_eq!(output.materialize_f64(), vec![0.0, 30.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn constructor_returns_object_with_properties() {
    let value = call(vec![row(&[10.0, 20.0, 40.0])]).unwrap();
    let Value::Object(obj) = value else {
        panic!("expected object");
    };
    assert!(obj.class_name().is(CLASS_IDENTITY));
    assert_eq!(
        obj.properties().get(METHOD),
        Some(&Value::String("linear".to_string().into()))
    );
    assert!(matches!(
        obj.properties().get(GRID_VECTORS),
        Some(Value::Cell(_))
    ));
}

#[test]
fn one_dimensional_linear_interpolation_and_none_extrapolation() {
    let obj = call(vec![
        col(&[0.0, 1.0, 2.0]),
        col(&[0.0, 10.0, 40.0]),
        Value::String("linear".to_string().into()),
        Value::String("none".to_string().into()),
    ])
    .unwrap();
    match subsref(obj, vec![row(&[-1.0, 0.5, 1.5, 3.0])])
        .unwrap()
        .value()
    {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[1, 4]);
            assert!(tensor.materialize_f64()[0].is_nan());
            assert_eq!(tensor.materialize_f64()[1], 5.0);
            assert_eq!(tensor.materialize_f64()[2], 25.0);
            assert!(tensor.materialize_f64()[3].is_nan());
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn one_dimensional_queries_preserve_rank_zero_and_rank_one_projection() {
    let interpolant = || {
        call(vec![col(&[0.0, 1.0, 2.0]), col(&[0.0, 10.0, 20.0])])
            .expect("one-dimensional interpolant")
    };
    let scalar = subsref(
        interpolant(),
        vec![Value::Tensor(Tensor::new(vec![0.5], Vec::new()).unwrap())],
    )
    .expect("rank-zero query");
    assert_eq!(scalar, Value::Num(5.0));

    let result = subsref(
        interpolant(),
        vec![Value::Tensor(Tensor::new(vec![0.5, 1.5], vec![2]).unwrap())],
    )
    .expect("rank-one query");
    let Value::Tensor(vector) = result.value() else {
        panic!("expected rank-one tensor");
    };
    assert_eq!(vector.shape(), &[2]);
    assert_eq!(vector.materialize_f64(), vec![5.0, 15.0]);
}

#[test]
fn default_grid_and_nearest_method_work() {
    let obj = call(vec![
        row(&[10.0, 20.0, 40.0]),
        Value::String("nearest".to_string().into()),
    ])
    .unwrap();
    match subsref(obj, vec![row(&[1.2, 2.6])]).unwrap().value() {
        Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![10.0, 40.0]),
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn single_values_reshape_and_interpolate_in_native_class() {
    let values = Tensor::from_f32(vec![0.0, 10.0, 40.0], vec![1, 3]).unwrap();
    let obj = call(vec![Value::Tensor(values)]).unwrap();
    subsref(obj, vec![row(&[1.5, 2.5])])
        .unwrap()
        .with_owned(|value| {
            let Value::Tensor(tensor) = value else {
                panic!("expected single tensor, got {value:?}");
            };
            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
            assert_eq!(
                tensor.into_numeric_storage().unwrap(),
                NumericStorage::F32(vec![5.0, 25.0])
            );
        });
}

#[test]
fn two_dimensional_grid_vectors_interpolate() {
    let grid =
        CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2]).unwrap();
    let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
    let obj = call(vec![Value::Cell(grid), Value::Tensor(values)]).unwrap();
    match subsref(obj, vec![Value::Num(0.5), Value::Num(1.0)])
        .unwrap()
        .value()
    {
        Value::Num(value) => assert_eq!(*value, 15.0),
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn two_dimensional_spline_uses_shared_not_a_knot_tensor_product() {
    let y = [-1.0, 0.0, 1.5, 3.0];
    let x = [0.0, 0.7, 2.0, 4.0];
    let polynomial = |x: f64, y: f64| x * x + 3.0 * x * y + 2.0 * y * y;
    let values: Vec<f64> = x
        .iter()
        .flat_map(|&col| y.iter().map(move |&row| polynomial(col, row)))
        .collect();
    let vectors = CellArray::new_with_shape(vec![col(&y), row(&x)], vec![1, 2]).unwrap();
    let obj = call(vec![
        Value::Cell(vectors),
        Value::Tensor(Tensor::new(values, vec![4, 4]).unwrap()),
        Value::String("spline".into()),
    ])
    .expect("2-D spline interpolant");
    let result = subsref(obj, vec![Value::Num(0.4), Value::Num(1.2)]).expect("2-D spline query");
    let Value::Num(actual) = result.value() else {
        panic!("double scalar expected")
    };
    assert!((*actual - polynomial(1.2, 0.4)).abs() < 1e-9);
}

#[test]
fn two_dimensional_spline_query_retains_exact_single_output_claim() {
    let vectors = CellArray::new_with_shape(
        vec![col(&[0.0, 1.0, 2.0, 3.0]), col(&[0.0, 1.0, 2.0, 3.0])],
        vec![1, 2],
    )
    .unwrap();
    let samples = Tensor::from_f32((0..16).map(|n| n as f32).collect(), vec![4, 4]).unwrap();
    let receiver = call(vec![
        Value::Cell(vectors),
        Value::Tensor(samples),
        Value::String("spline".into()),
    ])
    .unwrap();
    let ledger =
        crate::context::host_effect::ManagedHostEffectLedger::new_with_epoch(1_000_000, 19);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        crate::context::RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedSplineService(ledger.clone()))),
    );
    let queries = Value::Cell(
        CellArray::new_with_shape(vec![col(&[0.5, 1.5]), col(&[0.5, 1.5])], vec![1, 2]).unwrap(),
    );
    let subscript =
        crate::object::indexing::standard_substruct_fixture_from_parts(OBJECT_INDEX_PAREN, queries)
            .unwrap();
    let output = block_on(gridded_interpolant_subsref_registered(
        Some(runtime),
        receiver,
        subscript,
    ))
    .unwrap();
    let Value::Tensor(tensor) = output.value() else {
        panic!("single tensor expected")
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    assert_eq!(tensor.shape(), &[2, 1]);
    let retained = crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
        NumericDType::F32,
        &[2, 1],
    )
    .unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
    assert_eq!(ledger.snapshot().live_claimed_bytes, retained);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn full_query_grid_visits_column_major_points_without_changing_shape() {
    let grid =
        CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2]).unwrap();
    let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
    let obj = call(vec![Value::Cell(grid), Value::Tensor(values)]).unwrap();
    let queries =
        CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2]).unwrap();
    let result = subsref(obj, vec![Value::Cell(queries)]).unwrap();
    let Value::Tensor(tensor) = result.value() else {
        panic!("expected array result");
    };
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.materialize_f64(), vec![0.0, 10.0, 20.0, 30.0]);
}

#[test]
fn grid_vectors_values_and_queries_read_typed_integer_storage_exactly() {
    let grid = CellArray::new_with_shape(
        vec![
            int_col(IntegerStorage::I16(vec![0, 1])),
            int_col(IntegerStorage::U16(vec![0, 2])),
        ],
        vec![1, 2],
    )
    .unwrap();
    let values = int_tensor(IntegerStorage::I16(vec![0, 10, 20, 30]), vec![2, 2]);
    let obj = call(vec![Value::Cell(grid), values]).unwrap();
    let query = int_tensor(IntegerStorage::I16(vec![0, 1, 0, 2]), vec![2, 2]);
    match subsref(obj, vec![query]).unwrap().value() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.shape(), &[2, 1]);
            assert_eq!(tensor.materialize_f64(), vec![0.0, 30.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }

    let x_grid = int_tensor(IntegerStorage::I16(vec![0, 1, 0, 1]), vec![2, 2]);
    let y_grid = int_tensor(IntegerStorage::U16(vec![0, 0, 2, 2]), vec![2, 2]);
    let values = int_tensor(IntegerStorage::I16(vec![0, 10, 20, 30]), vec![2, 2]);
    let obj = call(vec![x_grid, y_grid, values]).unwrap();
    match subsref(obj, vec![Value::Num(1.0), Value::Num(2.0)])
        .unwrap()
        .value()
    {
        Value::Num(value) => assert_eq!(*value, 30.0),
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn one_dimensional_spline_uses_piecewise_polynomial_path() {
    let obj = call(vec![
        col(&[1.0, 2.0, 3.0]),
        int_col(IntegerStorage::I16(vec![1, 4, 9])),
        Value::String("spline".to_string().into()),
    ])
    .unwrap();
    match subsref(obj, vec![Value::Num(1.5)]).unwrap().value() {
        Value::Num(value) => assert!((value - 2.25).abs() < 1.0e-10),
        other => panic!("expected scalar, got {other:?}"),
    }
}

#[test]
fn one_dimensional_pchip_preserves_samples_and_nearest_extrapolates() {
    let obj = call(vec![
        col(&[1.0, 2.0, 3.0, 4.0]),
        int_col(IntegerStorage::I16(vec![0, 10, 15, 18])),
        Value::String("pchip".to_string().into()),
        Value::String("nearest".to_string().into()),
    ])
    .unwrap();
    match subsref(obj, vec![row(&[0.0, 3.0, 5.0])]).unwrap().value() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.materialize_f64()[0], 0.0);
            assert_eq!(tensor.materialize_f64()[1], 15.0);
            assert_eq!(tensor.materialize_f64()[2], 18.0);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn extrapolation_method_is_independent_from_interpolation_method() {
    let obj = call(vec![
        col(&[0.0, 1.0, 2.0]),
        col(&[0.0, 10.0, 40.0]),
        Value::String("nearest".to_string().into()),
        Value::String("linear".to_string().into()),
    ])
    .unwrap();
    match subsref(obj, vec![row(&[-1.0, 0.4, 3.0])]).unwrap().value() {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.materialize_f64()[0], -10.0);
            assert_eq!(tensor.materialize_f64()[1], 0.0);
            assert_eq!(tensor.materialize_f64()[2], 70.0);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn unsupported_cubic_method_is_explicit() {
    let err = call(vec![
        row(&[10.0, 20.0, 40.0]),
        Value::String("cubic".to_string().into()),
    ])
    .expect_err("cubic is not implemented yet");
    assert_eq!(
        err.identifier(),
        Some("RunMat:griddedInterpolant:UnsupportedMethod")
    );
}

#[test]
fn higher_order_methods_reject_nd_grids_explicitly() {
    let grid = CellArray::new_with_shape(
        vec![col(&[0.0, 1.0]), col(&[0.0, 2.0]), col(&[0.0, 3.0])],
        vec![1, 3],
    )
    .unwrap();
    let values = Tensor::new(
        vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        vec![2, 2, 2],
    )
    .unwrap();
    let err = call(vec![
        Value::Cell(grid),
        Value::Tensor(values),
        Value::String("spline".to_string().into()),
    ])
    .expect_err("three-dimensional spline is not implemented");
    assert_eq!(
        err.identifier(),
        Some("RunMat:griddedInterpolant:UnsupportedMethod")
    );
}

#[test]
fn pchip_requires_four_grid_points() {
    let err = call(vec![
        col(&[1.0, 2.0, 3.0]),
        col(&[1.0, 4.0, 9.0]),
        Value::String("pchip".to_string().into()),
    ])
    .expect_err("pchip requires four points");
    assert_eq!(
        err.identifier(),
        Some("RunMat:griddedInterpolant:InvalidInput")
    );
}

#[test]
fn full_grid_constructor_rejects_non_rectilinear_arrays() {
    let x_grid = Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
    let y_grid = Tensor::new(vec![0.0, 0.0, 2.0, 2.0], vec![2, 2]).unwrap();
    let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
    let err = call(vec![
        Value::Tensor(x_grid),
        Value::Tensor(y_grid),
        Value::Tensor(values),
    ])
    .expect_err("malformed full grid should be rejected");
    assert_eq!(
        err.identifier(),
        Some("RunMat:griddedInterpolant:InvalidInput")
    );
}

#[test]
fn rowwise_matrix_query_requires_two_dimensions() {
    let grid =
        CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 2.0])], vec![1, 2]).unwrap();
    let values = Tensor::new(vec![0.0, 10.0, 20.0, 30.0], vec![2, 2]).unwrap();
    let obj = call(vec![Value::Cell(grid), Value::Tensor(values)]).unwrap();
    let query = Tensor::new(vec![0.0, 0.0, 1.0, 2.0], vec![2, 2, 1]).unwrap();
    let err =
        subsref(obj, vec![Value::Tensor(query)]).expect_err("row-wise matrix query must be 2-D");
    assert_eq!(
        err.identifier(),
        Some("RunMat:griddedInterpolant:InvalidInput")
    );
}

#[test]
fn mutated_object_properties_are_revalidated_before_evaluation() {
    let value = call(vec![row(&[1.0, 2.0, 3.0])]).unwrap();
    let Value::Object(mut obj) = value else {
        panic!("expected object");
    };
    obj.replace_property_unbounded(
        VALUES.to_string(),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
    );
    let err = subsref(Value::Object(obj), vec![Value::Num(2.0)])
        .expect_err("invalid object state should be rejected");
    assert_eq!(
        err.identifier(),
        Some("RunMat:griddedInterpolant:InvalidInput")
    );
}

#[test]
fn member_subsref_returns_public_properties() {
    let obj = call(vec![row(&[1.0, 2.0, 3.0])]).unwrap();
    let value = block_on(gridded_interpolant_subsref(
        obj,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(VALUES.to_string().into()),
    ))
    .unwrap();
    assert!(matches!(value.value(), Value::Tensor(_)));
}

#[test]
fn one_dimensional_makima_preserves_class_and_rejects_nd() {
    let values = Tensor::from_f32(vec![0.0, 0.0, 0.0, 1.0, 0.0], vec![5, 1]).unwrap();
    let obj = call(vec![
        col(&[0.0, 1.0, 2.0, 3.0, 4.0]),
        Value::Tensor(values),
        Value::String("makima".to_string().into()),
    ])
    .expect("one-dimensional makima");
    let Value::Object(ref object) = obj else {
        panic!("expected interpolant object");
    };
    assert_eq!(
        object.properties().get(METHOD),
        Some(&Value::String("makima".to_string().into()))
    );
    assert_eq!(
        object.properties().get(EXTRAPOLATION_METHOD),
        Some(&Value::String("makima".to_string().into()))
    );
    let result = subsref(obj, vec![row(&[2.5, 4.0])]).expect("makima query");
    let Value::Tensor(tensor) = result.value() else {
        panic!("expected single tensor");
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    let actual = tensor.materialize_f64();
    assert!((actual[0] - 39.0 / 88.0).abs() < 1e-7);
    assert_eq!(actual[1], 0.0);

    let grid =
        CellArray::new_with_shape(vec![col(&[0.0, 1.0]), col(&[0.0, 1.0])], vec![1, 2]).unwrap();
    let values = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
    let err = call(vec![
        Value::Cell(grid),
        Value::Tensor(values),
        Value::String("makima".to_string().into()),
    ])
    .expect_err("N-D makima remains unavailable");
    assert_eq!(
        err.identifier(),
        Some("RunMat:griddedInterpolant:UnsupportedMethod")
    );
}
