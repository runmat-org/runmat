//! MATLAB-compatible `gobjects` builtin.

use runmat_builtins::{
    GOBJECTS_DESCRIPTOR, GOBJECTS_ERROR_ALLOCATION, GOBJECTS_ERROR_INVALID_SIZE,
    GOBJECTS_ERROR_TOO_MANY_OUTPUTS,
};

use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, NumericScalar, Tensor, Value};

#[runtime_builtin(
    name = "gobjects",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::plotting::gobjects"
)]
pub async fn gobjects_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    arguments: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    {
        let _scope = runtime.as_ref().map(|runtime| runtime.enter());
        crate::output_count::validate_fixed_outputs(
            &GOBJECTS_DESCRIPTOR,
            &GOBJECTS_ERROR_TOO_MANY_OUTPUTS,
            "gobjects",
        )
        .map_err(super::coordinate_admission::terminal)?;
    }
    let work = super::coordinate_admission::CoordinateWork::admit(runtime.as_ref(), "gobjects")?;
    let result = if let Some(runtime) = runtime.as_ref() {
        runtime.scope(gobjects_impl(arguments.view(), &work)).await
    } else {
        gobjects_impl(arguments.view(), &work).await
    };
    work.finish(result)
}

async fn gobjects_impl(
    args: crate::call::FunctionArgumentView<'_>,
    work: &super::coordinate_admission::CoordinateWork<'_>,
) -> crate::BuiltinResult<Value> {
    reject_unsupported_controls(args)?;
    let shape = parse_shape(args)?;
    let len = checked_element_count(&shape)?;
    let mut slots = Vec::new();
    slots.try_reserve_exact(len).map_err(|_| {
        crate::build_runtime_error(GOBJECTS_ERROR_ALLOCATION.message)
            .with_builtin("gobjects")
            .with_identifier(GOBJECTS_ERROR_ALLOCATION.identifier.unwrap())
            .build()
    })?;
    slots.resize(len, f64::NAN);
    work.check()?;
    let tensor = Tensor::new_with_dtype(slots, shape, NumericDType::F64).map_err(|source| {
        crate::build_runtime_error(format!("gobjects: {source}"))
            .with_builtin("gobjects")
            .with_identifier(GOBJECTS_ERROR_ALLOCATION.identifier.unwrap())
            .build()
    })?;
    Ok(Value::Tensor(tensor))
}

fn reject_unsupported_controls(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
    if args.iter().any(|arg| {
        matches!(
            arg,
            Value::Bool(_) | Value::LogicalArray(_) | Value::GpuTensor(_)
        )
    }) {
        return Err(
            crate::build_runtime_error(GOBJECTS_ERROR_INVALID_SIZE.message)
                .with_builtin("gobjects")
                .with_identifier(GOBJECTS_ERROR_INVALID_SIZE.identifier.unwrap())
                .build(),
        );
    }
    Ok(())
}

fn parse_shape(args: crate::call::FunctionArgumentView<'_>) -> crate::BuiltinResult<Vec<usize>> {
    if args.is_empty() {
        return Ok(vec![1, 1]);
    }
    if args.len() == 1 {
        let dimensions = dimensions_from_single_argument(&args[0])?;
        return Ok(match dimensions.len() {
            0 => vec![0, 0],
            1 => vec![dimensions[0], dimensions[0]],
            _ => dimensions,
        });
    }
    args.iter().map(parse_scalar_size).collect()
}

fn dimensions_from_single_argument(value: &Value) -> crate::BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(_) | Value::Int(_) => Ok(vec![parse_scalar_size(value)?]),
        Value::Tensor(tensor) if tensor.len() == 1 => Ok(vec![parse_scalar_size(value)?]),
        Value::Tensor(tensor) if tensor.shape().len() == 2 && tensor.shape()[0] == 1 => (0..tensor
            .len())
            .map(|index| {
                parse_numeric_scalar_size(
                    tensor
                        .numeric_value_at(index)
                        .expect("index within gobjects size vector"),
                )
            })
            .collect(),
        _ => Err(invalid_size_error(
            "a single nonscalar size argument must be a numeric row vector",
        )),
    }
}

fn parse_scalar_size(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::Num(number) => parse_float_size(*number),
        Value::Int(integer) => parse_integer_size(integer),
        Value::Tensor(tensor) if tensor.len() == 1 => parse_numeric_scalar_size(
            tensor
                .numeric_value_at(0)
                .expect("one-element gobjects size tensor"),
        ),
        _ => Err(invalid_size_error("size arguments must be numeric scalars")),
    }
}

fn parse_numeric_scalar_size(value: NumericScalar) -> crate::BuiltinResult<usize> {
    match value {
        NumericScalar::F64(value) => parse_float_size(value),
        NumericScalar::F32(value) => parse_float_size(f64::from(value)),
        value => parse_integer_size(
            &value
                .into_int_value()
                .expect("nonfloating numeric scalar is integer"),
        ),
    }
}

fn parse_integer_size(value: &IntValue) -> crate::BuiltinResult<usize> {
    let negative = match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        _ => false,
    };
    if negative {
        return Ok(0);
    }
    value
        .try_to_usize()
        .ok_or_else(|| invalid_size_error("integer size exceeds the platform dimension range"))
}

fn parse_float_size(value: f64) -> crate::BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(invalid_size_error("sizes must be finite integer values"));
    }
    if value < 0.0 {
        return Ok(0);
    }
    if value >= 2.0_f64.powi(usize::BITS as i32) {
        return Err(invalid_size_error(
            "size exceeds the platform dimension range",
        ));
    }
    Ok(value as usize)
}

fn checked_element_count(shape: &[usize]) -> crate::BuiltinResult<usize> {
    let count = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
    let Some(count) = count else {
        return Err(allocation_error());
    };
    if count > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(allocation_error());
    }
    Ok(count)
}

fn invalid_size_error(detail: &str) -> crate::RuntimeError {
    crate::build_runtime_error(format!("{}: {detail}", GOBJECTS_ERROR_INVALID_SIZE.message))
        .with_builtin("gobjects")
        .with_identifier(GOBJECTS_ERROR_INVALID_SIZE.identifier.unwrap())
        .build()
}

fn allocation_error() -> crate::RuntimeError {
    crate::build_runtime_error(GOBJECTS_ERROR_ALLOCATION.message)
        .with_builtin("gobjects")
        .with_identifier(GOBJECTS_ERROR_ALLOCATION.identifier.unwrap())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_builtins::{
        GOBJECTS_DESCRIPTOR, GOBJECTS_ERROR_ALLOCATION, GOBJECTS_ERROR_INVALID_SIZE,
        GOBJECTS_INTEGER_CAPABILITIES,
    };

    fn tensor_from(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn gobjects_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = GOBJECTS_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"h = gobjects()"));
        assert!(labels.contains(&"h = gobjects(sz...)"));
    }

    #[test]
    fn gobjects_defaults_to_scalar_placeholder() {
        let tensor = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .unwrap(),
        );
        assert_eq!(tensor.shape(), &[1, 1]);
        assert_eq!(tensor.materialize_f64().len(), 1);
        assert!(tensor.materialize_f64()[0].is_nan());
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ))
        .with_cancellation_token(source.token());
        let arguments = crate::call::BuiltinArguments::from_borrowed(&[], Some(&runtime)).unwrap();
        source.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(gobjects_builtin(Some(runtime), arguments)).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error.identifier().unwrap().contains("Cancelled"));

        let _outputs = crate::output_count::push_output_count(Some(2)).unwrap();
        let error = block_on(gobjects_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Bool(true)]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:gobjects:TooManyOutputs"));
    }

    #[test]
    fn gobjects_accepts_scalar_dims_and_size_vector() {
        let square = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
            ))
            .unwrap(),
        );
        assert_eq!(square.shape(), &[3, 3]);
        assert!(square.materialize_f64().iter().all(|value| value.is_nan()));

        let rect = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Num(2.0),
                    Value::Num(3.0),
                ]),
            ))
            .unwrap(),
        );
        assert_eq!(rect.shape(), &[2, 3]);

        let size_vec = Tensor::new(vec![4.0, 1.0], vec![1, 2]).unwrap();
        let vector = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                    size_vec,
                )]),
            ))
            .unwrap(),
        );
        assert_eq!(vector.shape(), &[4, 1]);
    }

    #[test]
    fn gobjects_rejects_invalid_dimensions() {
        let negative = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(-1.0)]),
            ))
            .unwrap(),
        );
        assert_eq!(negative.shape(), &[0, 0]);
        assert!(block_on(gobjects_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.5)])
        ))
        .is_err());
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(block_on(gobjects_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix)])
        ))
        .is_err());
    }

    #[test]
    fn gobjects_accepts_every_documented_integer_size_class_exactly() {
        assert_eq!(GOBJECTS_INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
        for integer in [
            IntValue::I8(2),
            IntValue::I16(2),
            IntValue::I32(2),
            IntValue::I64(2),
            IntValue::U8(2),
            IntValue::U16(2),
            IntValue::U32(2),
            IntValue::U64(2),
        ] {
            let tensor = tensor_from(
                block_on(gobjects_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                        integer,
                    )]),
                ))
                .unwrap(),
            );
            assert_eq!(tensor.shape(), &[2, 2]);
        }
    }

    #[test]
    fn gobjects_clamps_negative_signed_dimensions_to_zero() {
        for integer in [
            IntValue::I8(-1),
            IntValue::I16(-1),
            IntValue::I32(-1),
            IntValue::I64(-1),
        ] {
            let tensor = tensor_from(
                block_on(gobjects_builtin(
                    None,
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(
                        integer,
                    )]),
                ))
                .unwrap(),
            );
            assert_eq!(tensor.shape(), &[0, 0]);
        }
        let tensor = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::Int(IntValue::I16(-2)),
                    Value::Num(3.0),
                ]),
            ))
            .unwrap(),
        );
        assert_eq!(tensor.shape(), &[0, 3]);
    }

    #[test]
    fn gobjects_requires_row_vector_geometry_for_vector_sizes() {
        let row = Tensor::from_numeric_storage(
            runmat_value::NumericStorage::F32(vec![2.0, 3.0]),
            vec![1, 2],
        )
        .expect("row");
        let tensor = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(row)]),
            ))
            .unwrap(),
        );
        assert_eq!(tensor.shape(), &[2, 3]);

        let column = Tensor::new(vec![2.0, 3.0], vec![2, 1]).expect("column");
        let error = block_on(gobjects_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(column)]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), GOBJECTS_ERROR_INVALID_SIZE.identifier);

        let vector = Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("vector");
        assert!(block_on(gobjects_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(vector),
                Value::Num(4.0)
            ])
        ))
        .is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gobjects_low_rank_size_inputs_keep_scalar_before_row_vector_admission() {
        let scalar = Tensor::new(vec![2.0], Vec::new()).expect("rank-zero scalar size");
        let output = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                    scalar,
                )]),
            ))
            .unwrap(),
        );
        assert_eq!(output.shape(), &[2, 2]);

        let rank_one = Tensor::new(vec![2.0, 3.0], vec![2]).expect("rank-one sizes");
        let error = block_on(gobjects_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(rank_one)]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), GOBJECTS_ERROR_INVALID_SIZE.identifier);

        let row = Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("row sizes");
        let output = tensor_from(
            block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(row)]),
            ))
            .unwrap(),
        );
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn gobjects_rejects_logical_and_resident_controls_before_provider_access() {
        for value in [
            Value::Bool(true),
            Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![1], vec![1, 1]).expect("logical"),
            ),
            Value::GpuTensor(
                runmat_accelerate_api::test_support::TestAllocationOwner::new()
                    .allocate_zeroed(
                        vec![1, 1],
                        runmat_accelerate_api::NumericElementType::F64,
                        runmat_accelerate_api::GpuTensorStorage::Real,
                    )
                    .unwrap()
                    .handle,
            ),
        ] {
            let error = block_on(gobjects_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![value]),
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), GOBJECTS_ERROR_INVALID_SIZE.identifier);
        }
    }

    #[test]
    fn gobjects_checks_shape_products_before_allocating_placeholders() {
        let error = checked_element_count(&[usize::MAX, 2]).unwrap_err();
        assert_eq!(error.identifier(), GOBJECTS_ERROR_ALLOCATION.identifier);
        assert!(GOBJECTS_DESCRIPTOR.signatures[0].outputs[0]
            .description
            .contains("NaN numeric handle slots"));
    }

    #[test]
    fn gobjects_rejects_floating_dimension_at_platform_exclusive_bound() {
        let exclusive = 2.0_f64.powi(usize::BITS as i32);
        let error = block_on(gobjects_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Num(exclusive),
                Value::Num(0.0),
            ]),
        ))
        .expect_err("out-of-range dimension is invalid even when the product is zero");
        assert_eq!(error.identifier(), GOBJECTS_ERROR_INVALID_SIZE.identifier);
    }
}
