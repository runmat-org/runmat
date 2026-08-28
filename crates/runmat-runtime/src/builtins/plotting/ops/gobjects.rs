//! MATLAB-compatible `gobjects` builtin.

use crate::builtins::plotting::type_resolvers::handle_array_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, NumericDType, NumericScalar, Tensor, Value};

const GOBJECTS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Graphics-placeholder array; RunMat currently uses NaN numeric handle slots as its bounded representation.",
}];

const GOBJECTS_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const GOBJECTS_INPUTS_DIMS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sz",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Output dimensions, either as separate scalar sizes or a size vector.",
}];

const GOBJECTS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "h = gobjects()",
        inputs: &GOBJECTS_INPUTS_NONE,
        outputs: &GOBJECTS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "h = gobjects(sz...)",
        inputs: &GOBJECTS_INPUTS_DIMS,
        outputs: &GOBJECTS_OUTPUT,
    },
];

const GOBJECTS_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GOBJECTS.INVALID_SIZE",
    identifier: Some("RunMat:gobjects:InvalidSize"),
    when: "A size is nonnumeric, nonintegral, logical, resident, or has invalid vector geometry.",
    message: "gobjects: invalid size argument",
};
const GOBJECTS_ERROR_ALLOCATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GOBJECTS.ALLOCATION",
    identifier: Some("RunMat:gobjects:AllocationLimit"),
    when: "The requested shape exceeds addressable or allocatable placeholder storage.",
    message: "gobjects: requested placeholder array is too large",
};
const GOBJECTS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [GOBJECTS_ERROR_INVALID_SIZE, GOBJECTS_ERROR_ALLOCATION];

const GOBJECTS_INTEGER_SIZE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "size",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The compatibility target documents all eight integer classes plus integral single and double sizes. Negative signed dimensions are treated as zero.",
    }];
pub const GOBJECTS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "H = gobjects(integer_sizes...)",
        inputs: &GOBJECTS_INTEGER_SIZE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Scalar input creates a square shape; two or more scalar inputs create those dimensions; one row-vector input supplies dimensions exactly. RunMat retains a documented representation gap: NaN numeric slots stand in for GraphicsPlaceholder objects.",
    }];

pub const GOBJECTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GOBJECTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GOBJECTS_ERRORS,
};

#[runtime_builtin(
    name = "gobjects",
    category = "plotting",
    summary = "Create an array of graphics-handle placeholders.",
    keywords = "gobjects,graphics,handle,preallocate,plotting",
    suppress_auto_output = true,
    type_resolver(handle_array_type),
    descriptor(crate::builtins::plotting::gobjects::GOBJECTS_DESCRIPTOR),
    integer_capabilities(crate::builtins::plotting::gobjects::GOBJECTS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::plotting::gobjects"
)]
pub async fn gobjects_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    reject_unsupported_controls(&args)?;
    let shape = parse_shape(&args)?;
    let len = checked_element_count(&shape)?;
    let mut slots = Vec::new();
    slots.try_reserve_exact(len).map_err(|_| {
        crate::build_runtime_error(GOBJECTS_ERROR_ALLOCATION.message)
            .with_builtin("gobjects")
            .with_identifier(GOBJECTS_ERROR_ALLOCATION.identifier.unwrap())
            .build()
    })?;
    slots.resize(len, f64::NAN);
    let tensor = Tensor::new_with_dtype(slots, shape, NumericDType::F64)
        .map_err(|err| format!("gobjects: {err}"))?;
    Ok(Value::Tensor(tensor))
}

fn reject_unsupported_controls(args: &[Value]) -> crate::BuiltinResult<()> {
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

fn parse_shape(args: &[Value]) -> crate::BuiltinResult<Vec<usize>> {
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
        Value::Tensor(tensor) if tensor.shape.len() == 2 && tensor.shape[0] == 1 => (0..tensor
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
        let tensor = tensor_from(block_on(gobjects_builtin(Vec::new())).unwrap());
        assert_eq!(tensor.shape, vec![1, 1]);
        assert_eq!(tensor.materialize_f64().len(), 1);
        assert!(tensor.materialize_f64()[0].is_nan());
    }

    #[test]
    fn gobjects_accepts_scalar_dims_and_size_vector() {
        let square = tensor_from(block_on(gobjects_builtin(vec![Value::Num(3.0)])).unwrap());
        assert_eq!(square.shape, vec![3, 3]);
        assert!(square.materialize_f64().iter().all(|value| value.is_nan()));

        let rect = tensor_from(
            block_on(gobjects_builtin(vec![Value::Num(2.0), Value::Num(3.0)])).unwrap(),
        );
        assert_eq!(rect.shape, vec![2, 3]);

        let size_vec = Tensor::new(vec![4.0, 1.0], vec![1, 2]).unwrap();
        let vector =
            tensor_from(block_on(gobjects_builtin(vec![Value::Tensor(size_vec)])).unwrap());
        assert_eq!(vector.shape, vec![4, 1]);
    }

    #[test]
    fn gobjects_rejects_invalid_dimensions() {
        let negative = tensor_from(block_on(gobjects_builtin(vec![Value::Num(-1.0)])).unwrap());
        assert_eq!(negative.shape, vec![0, 0]);
        assert!(block_on(gobjects_builtin(vec![Value::Num(1.5)])).is_err());
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(block_on(gobjects_builtin(vec![Value::Tensor(matrix)])).is_err());
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
            let tensor =
                tensor_from(block_on(gobjects_builtin(vec![Value::Int(integer)])).unwrap());
            assert_eq!(tensor.shape, vec![2, 2]);
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
            let tensor =
                tensor_from(block_on(gobjects_builtin(vec![Value::Int(integer)])).unwrap());
            assert_eq!(tensor.shape, vec![0, 0]);
        }
        let tensor = tensor_from(
            block_on(gobjects_builtin(vec![
                Value::Int(IntValue::I16(-2)),
                Value::Num(3.0),
            ]))
            .unwrap(),
        );
        assert_eq!(tensor.shape, vec![0, 3]);
    }

    #[test]
    fn gobjects_requires_row_vector_geometry_for_vector_sizes() {
        let row = Tensor::from_numeric_storage(
            runmat_value::NumericStorage::F32(vec![2.0, 3.0]),
            vec![1, 2],
        )
        .expect("row");
        let tensor = tensor_from(block_on(gobjects_builtin(vec![Value::Tensor(row)])).unwrap());
        assert_eq!(tensor.shape, vec![2, 3]);

        let column = Tensor::new(vec![2.0, 3.0], vec![2, 1]).expect("column");
        let error = block_on(gobjects_builtin(vec![Value::Tensor(column)])).unwrap_err();
        assert_eq!(error.identifier(), GOBJECTS_ERROR_INVALID_SIZE.identifier);

        let vector = Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("vector");
        assert!(block_on(gobjects_builtin(vec![
            Value::Tensor(vector),
            Value::Num(4.0)
        ]))
        .is_err());
    }

    #[test]
    fn gobjects_rejects_logical_and_resident_controls_before_provider_access() {
        for value in [
            Value::Bool(true),
            Value::LogicalArray(
                runmat_value::LogicalArray::new(vec![1], vec![1, 1]).expect("logical"),
            ),
            Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                shape: vec![1, 1],
                device_id: 999_991,
                buffer_id: 999_991,
                descriptor: Default::default(),
            }),
        ] {
            let error = block_on(gobjects_builtin(vec![value])).unwrap_err();
            assert_eq!(error.identifier(), GOBJECTS_ERROR_INVALID_SIZE.identifier);
        }
    }

    #[test]
    fn gobjects_checks_shape_products_before_allocating_placeholders() {
        let error = checked_element_count(&[usize::MAX, 2]).unwrap_err();
        assert_eq!(error.identifier(), GOBJECTS_ERROR_ALLOCATION.identifier);
        assert!(GOBJECTS_OUTPUT[0]
            .description
            .contains("NaN numeric handle slots"));
    }

    #[test]
    fn gobjects_rejects_floating_dimension_at_platform_exclusive_bound() {
        let exclusive = 2.0_f64.powi(usize::BITS as i32);
        let error = block_on(gobjects_builtin(vec![
            Value::Num(exclusive),
            Value::Num(0.0),
        ]))
        .expect_err("out-of-range dimension is invalid even when the product is zero");
        assert_eq!(error.identifier(), GOBJECTS_ERROR_INVALID_SIZE.identifier);
    }
}
