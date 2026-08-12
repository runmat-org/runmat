//! Class-static empty-array constructors plus a compatibility-gated RunMat shorthand.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult};

const LABEL: &str = "empty";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmptyClass {
    Double,
    Single,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Logical,
    Char,
    String,
    Cell,
    Struct,
    GpuArray,
}

fn empty_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message).with_builtin(LABEL).build()
}

fn empty_descriptor_error(
    descriptor: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(LABEL);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

const EMPTY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Empty array preserving the receiver class.",
}];
const EMPTY_NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const EMPTY_DIMS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "dimensions",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Integer scalar dimensions; at least one dimension must be zero.",
}];
const EMPTY_SIZE_VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sizeVector",
    ty: BuiltinParamType::SizeArg,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of integer dimensions; at least one dimension must be zero.",
}];
const EMPTY_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "A = ClassName.empty",
        inputs: &EMPTY_NO_INPUTS,
        outputs: &EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ClassName.empty(sz1, ..., szN)",
        inputs: &EMPTY_DIMS_INPUTS,
        outputs: &EMPTY_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "A = ClassName.empty(sizeVector)",
        inputs: &EMPTY_SIZE_VECTOR_INPUTS,
        outputs: &EMPTY_OUTPUT,
    },
];
const EMPTY_ERRORS: [BuiltinErrorDescriptor; 4] = [
    BuiltinErrorDescriptor {
        code: "RM.EMPTY.INVALID_DIMS",
        identifier: Some("RunMat:empty:InvalidSize"),
        when: "Size arguments are not integer-valued numeric scalars or one row size vector.",
        message: "empty: invalid size arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.EMPTY.NON_EMPTY_SHAPE",
        identifier: Some("RunMat:empty:NonEmptyShape"),
        when: "Every normalized dimension is nonzero.",
        message: "empty: at least one dimension must be zero",
    },
    BuiltinErrorDescriptor {
        code: "RM.EMPTY.UNSUPPORTED_CLASS",
        identifier: Some("RunMat:empty:UnsupportedClass"),
        when: "The selected shorthand class cannot be represented as a shaped empty value.",
        message: "empty: unsupported output class",
    },
    BuiltinErrorDescriptor {
        code: "RM.EMPTY.INTERNAL",
        identifier: Some("RunMat:empty:Internal"),
        when: "The empty value cannot be assembled.",
        message: "empty: internal error",
    },
];

pub const EMPTY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &EMPTY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &EMPTY_ERRORS,
};

const EMPTY_GLOBAL_CALL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "empty-global-call",
    mode: BuiltinExtensionMode::RunMatOnly,
    description:
        "calling empty as a global function is a RunMat extension; MATLAB uses ClassName.empty",
    error_identifier: Some("RunMat:compatibility:EmptyGlobalCallExtension"),
};
const EMPTY_TYPENAME_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "empty-typename",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "selecting the empty output class with a trailing typename is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EmptyTypenameExtension"),
};
const EMPTY_RESIDENT_SIZE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "empty-resident-size",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "empty with a resident size control is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:EmptyResidentSizeExtension"),
};
pub const EMPTY_GLOBAL_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    EMPTY_GLOBAL_CALL_EXTENSION,
    EMPTY_TYPENAME_EXTENSION,
    EMPTY_RESIDENT_SIZE_EXTENSION,
];
pub const EMPTY_STATIC_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [EMPTY_RESIDENT_SIZE_EXTENSION];

const EMPTY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "IntegerClass receiver",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Each of the eight integer primitive classes exposes ClassName.empty and determines the exact output storage class.",
    },
    BuiltinIntegerInputCapability {
        name: "size dimensions",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Integer scalar dimensions and row size vectors are decoded from authoritative storage; signed negatives clamp to zero.",
    },
];
pub const EMPTY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "A = IntegerClass.empty, IntegerClass.empty(sz1, ..., szN), or IntegerClass.empty(sizeVector)",
        inputs: &EMPTY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Output is empty authoritative storage of the receiver class. At least one normalized dimension is zero, trailing singleton dimensions after dimension two are removed, and resident controls are independently gated before exact gather.",
    }];

#[runtime_builtin(
    name = "empty",
    category = "array/creation",
    summary = "Construct an empty array; global use is a RunMat-only shorthand for ClassName.empty.",
    keywords = "empty,preallocate,array",
    accel = "none",
    descriptor(crate::builtins::array::creation::empty::EMPTY_DESCRIPTOR),
    extensions(crate::builtins::array::creation::empty::EMPTY_GLOBAL_EXTENSIONS),
    integer_capabilities(crate::builtins::array::creation::empty::EMPTY_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::empty"
)]
async fn empty_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let (class, size_args, had_typename) = split_class_arg(args)?;
    let shape = parse_shape(&size_args).await?;
    crate::compatibility::ensure_builtin_extension_enabled(&EMPTY_GLOBAL_CALL_EXTENSION, LABEL)?;
    if had_typename {
        crate::compatibility::ensure_builtin_extension_enabled(&EMPTY_TYPENAME_EXTENSION, LABEL)?;
    }
    build_empty(class, shape).await
}

macro_rules! register_integer_empty {
    ($rust_name:ident, $builtin_name:literal, $class:expr) => {
        #[runtime_builtin(
            name = $builtin_name,
            category = "array/creation",
            summary = "Construct an empty integer array preserving the class receiver.",
            keywords = "empty,integer,preallocate,array",
            accel = "none",
            descriptor(crate::builtins::array::creation::empty::EMPTY_DESCRIPTOR),
            extensions(crate::builtins::array::creation::empty::EMPTY_STATIC_EXTENSIONS),
            integer_capabilities(crate::builtins::array::creation::empty::EMPTY_INTEGER_CAPABILITIES),
            builtin_path = "crate::builtins::array::creation::empty"
        )]
        async fn $rust_name(args: Vec<Value>) -> BuiltinResult<Value> {
            let shape = parse_shape(&args).await?;
            build_empty($class, shape).await
        }
    };
}

register_integer_empty!(int8_empty_builtin, "int8.empty", EmptyClass::Int8);
register_integer_empty!(int16_empty_builtin, "int16.empty", EmptyClass::Int16);
register_integer_empty!(int32_empty_builtin, "int32.empty", EmptyClass::Int32);
register_integer_empty!(int64_empty_builtin, "int64.empty", EmptyClass::Int64);
register_integer_empty!(uint8_empty_builtin, "uint8.empty", EmptyClass::Uint8);
register_integer_empty!(uint16_empty_builtin, "uint16.empty", EmptyClass::Uint16);
register_integer_empty!(uint32_empty_builtin, "uint32.empty", EmptyClass::Uint32);
register_integer_empty!(uint64_empty_builtin, "uint64.empty", EmptyClass::Uint64);

fn class_from_keyword(keyword: &str) -> Option<EmptyClass> {
    Some(match keyword {
        "double" => EmptyClass::Double,
        "single" => EmptyClass::Single,
        "int8" => EmptyClass::Int8,
        "int16" => EmptyClass::Int16,
        "int32" => EmptyClass::Int32,
        "int64" => EmptyClass::Int64,
        "uint8" => EmptyClass::Uint8,
        "uint16" => EmptyClass::Uint16,
        "uint32" => EmptyClass::Uint32,
        "uint64" => EmptyClass::Uint64,
        "logical" => EmptyClass::Logical,
        "char" => EmptyClass::Char,
        "string" => EmptyClass::String,
        "cell" => EmptyClass::Cell,
        "struct" => EmptyClass::Struct,
        "gpuarray" => EmptyClass::GpuArray,
        _ => return None,
    })
}

fn split_class_arg(mut args: Vec<Value>) -> BuiltinResult<(EmptyClass, Vec<Value>, bool)> {
    if args.iter().any(|arg| {
        keyword_of(arg)
            .as_deref()
            .is_some_and(|keyword| keyword == "like")
    }) {
        return Err(empty_error(
            "empty: the global 'like' shorthand is not supported; use ClassName.empty",
        ));
    }
    if let Some(class) = args
        .last()
        .and_then(keyword_of)
        .as_deref()
        .and_then(class_from_keyword)
    {
        args.pop();
        return Ok((class, args, true));
    }
    Ok((EmptyClass::Double, args, false))
}

async fn parse_shape(args: &[Value]) -> BuiltinResult<Vec<usize>> {
    let mut shape = if args.is_empty() {
        vec![0, 0]
    } else if args.len() == 1 {
        let host = host_size_value(&args[0]).await?;
        parse_single_size_argument(&host)?
    } else {
        let mut dims = Vec::with_capacity(args.len());
        for arg in args {
            let host = host_size_value(arg).await?;
            dims.push(parse_size_scalar(&host)?);
        }
        dims
    };
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    if shape.len() == 1 {
        shape.push(shape[0]);
    }
    if !shape.contains(&0) {
        return Err(empty_descriptor_error(
            &EMPTY_ERRORS[1],
            "empty: at least one dimension must be zero to construct an empty array",
        ));
    }
    Ok(shape)
}

async fn host_size_value(value: &Value) -> BuiltinResult<Value> {
    if let Value::GpuTensor(handle) = value {
        validate_row_or_scalar_shape(&handle.shape)?;
        crate::compatibility::ensure_builtin_extension_enabled(
            &EMPTY_RESIDENT_SIZE_EXTENSION,
            LABEL,
        )?;
        return gather_if_needed_async(value)
            .await
            .map_err(|err| empty_error(format!("empty: failed to gather size control: {err}")));
    }
    Ok(value.clone())
}

fn parse_single_size_argument(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(_) | Value::Num(_) => {
            let dim = parse_size_scalar(value)?;
            Ok(vec![dim, dim])
        }
        Value::Tensor(tensor) => parse_size_tensor(tensor),
        other => Err(empty_error(format!(
            "empty: size input must be a numeric scalar or row vector, got {other:?}"
        ))),
    }
}

fn parse_size_tensor(value: &Tensor) -> BuiltinResult<Vec<usize>> {
    if value.is_empty() {
        return Ok(vec![0, 0]);
    }
    validate_row_or_scalar_shape(&value.shape)?;
    if value.len() == 1 {
        let dim = parse_size_scalar(&Value::Tensor(value.clone()))?;
        return Ok(vec![dim, dim]);
    }
    if let Some(storage) = value.integer_storage() {
        return (0..storage.len())
            .map(|index| {
                storage
                    .value_at(index)
                    .ok_or_else(|| empty_error("empty: inconsistent integer size storage"))
                    .and_then(|value| parse_integer_dimension(&value))
            })
            .collect();
    }
    (0..value.len())
        .map(|index| parse_numeric_dimension(tensor::tensor_value_f64(value, index)))
        .collect()
}

fn validate_row_or_scalar_shape(shape: &[usize]) -> BuiltinResult<()> {
    let valid = match shape.len() {
        0 | 1 => true,
        2 => shape[0] <= 1,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(empty_error("empty: size vector must be a row vector"))
    }
}

fn parse_size_scalar(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(value) => parse_integer_dimension(value),
        Value::Num(value) => parse_numeric_dimension(*value),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                parse_integer_dimension(&value)
            } else {
                parse_numeric_dimension(tensor::tensor_value_f64(tensor, 0))
            }
        }
        other => Err(empty_error(format!(
            "empty: each dimension must be a numeric scalar, got {other:?}"
        ))),
    }
}

fn parse_integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    let result = match value {
        IntValue::I8(value) => usize::try_from((*value).max(0) as u64),
        IntValue::I16(value) => usize::try_from((*value).max(0) as u64),
        IntValue::I32(value) => usize::try_from((*value).max(0) as u64),
        IntValue::I64(value) => usize::try_from((*value).max(0) as u64),
        IntValue::U8(value) => usize::try_from(*value as u64),
        IntValue::U16(value) => usize::try_from(*value as u64),
        IntValue::U32(value) => usize::try_from(*value as u64),
        IntValue::U64(value) => usize::try_from(*value),
    };
    result.map_err(|_| empty_error("empty: dimension exceeds the supported platform range"))
}

fn parse_numeric_dimension(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(empty_error("empty: dimensions must be finite integers"));
    }
    if value < 0.0 {
        return Ok(0);
    }
    let fits_usize = value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64);
    if value.fract() != 0.0 || !fits_usize {
        return Err(empty_error(
            "empty: dimension must be an integer within the supported platform range",
        ));
    }
    Ok(value as usize)
}

async fn build_empty(class: EmptyClass, shape: Vec<usize>) -> BuiltinResult<Value> {
    match class {
        EmptyClass::Double => Tensor::new(Vec::new(), shape)
            .map(tensor::tensor_into_value)
            .map_err(empty_error),
        EmptyClass::Single => Tensor::new_with_dtype(Vec::new(), shape, NumericDType::F32)
            .map(tensor::tensor_into_value)
            .map_err(empty_error),
        EmptyClass::Int8 => integer_empty(IntegerStorage::I8(Vec::new()), shape),
        EmptyClass::Int16 => integer_empty(IntegerStorage::I16(Vec::new()), shape),
        EmptyClass::Int32 => integer_empty(IntegerStorage::I32(Vec::new()), shape),
        EmptyClass::Int64 => integer_empty(IntegerStorage::I64(Vec::new()), shape),
        EmptyClass::Uint8 => integer_empty(IntegerStorage::U8(Vec::new()), shape),
        EmptyClass::Uint16 => integer_empty(IntegerStorage::U16(Vec::new()), shape),
        EmptyClass::Uint32 => integer_empty(IntegerStorage::U32(Vec::new()), shape),
        EmptyClass::Uint64 => integer_empty(IntegerStorage::U64(Vec::new()), shape),
        EmptyClass::Logical => Ok(Value::LogicalArray(LogicalArray::zeros(shape))),
        EmptyClass::Char => {
            if shape.len() != 2 {
                return Err(empty_error(
                    "empty: character arrays must be two-dimensional",
                ));
            }
            CharArray::new(Vec::new(), shape[0], shape[1])
                .map(Value::CharArray)
                .map_err(empty_error)
        }
        EmptyClass::String => StringArray::new(Vec::new(), shape)
            .map(Value::StringArray)
            .map_err(empty_error),
        EmptyClass::Cell => make_cell_with_shape(Vec::new(), shape).map_err(empty_error),
        EmptyClass::Struct => Err(empty_error(
            "empty: shaped struct empties require first-class struct-array storage",
        )),
        EmptyClass::GpuArray => {
            let mut zeros_args = shape
                .iter()
                .map(|dimension| Value::from(*dimension as f64))
                .collect::<Vec<_>>();
            zeros_args.push(Value::from("gpuArray"));
            crate::call_builtin_async("zeros", &zeros_args)
                .await
                .map_err(Into::into)
        }
    }
}

fn integer_empty(storage: IntegerStorage, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new_integer(storage, shape)
        .map(tensor::tensor_into_value)
        .map_err(empty_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_static(class: EmptyClass, args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(async {
            let shape = parse_shape(&args).await?;
            build_empty(class, shape).await
        })
    }

    #[test]
    fn all_integer_receivers_preserve_exact_empty_storage() {
        for (class, dtype) in [
            (EmptyClass::Int8, NumericDType::I8),
            (EmptyClass::Int16, NumericDType::I16),
            (EmptyClass::Int32, NumericDType::I32),
            (EmptyClass::Int64, NumericDType::I64),
            (EmptyClass::Uint8, NumericDType::U8),
            (EmptyClass::Uint16, NumericDType::U16),
            (EmptyClass::Uint32, NumericDType::U32),
            (EmptyClass::Uint64, NumericDType::U64),
        ] {
            let Value::Tensor(value) = run_static(
                class,
                vec![Value::Int(IntValue::U64(0)), Value::Int(IntValue::U64(7))],
            )
            .unwrap() else {
                panic!("expected tensor");
            };
            assert_eq!(value.shape, vec![0, 7]);
            assert_eq!(value.numeric_dtype(), dtype);
            assert!(value.integer_storage().is_some());
            assert!(value.is_empty());
        }
    }

    #[test]
    fn signed_negative_dimensions_clamp_and_trailing_ones_trim() {
        let Value::Tensor(value) = run_static(
            EmptyClass::Int64,
            vec![
                Value::Int(IntValue::I64(-9)),
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::U8(1)),
                Value::Int(IntValue::U8(1)),
            ],
        )
        .unwrap() else {
            panic!("expected tensor");
        };
        assert_eq!(value.shape, vec![0, usize::MAX]);
        assert_eq!(value.numeric_dtype(), NumericDType::I64);
    }

    #[test]
    fn row_size_vectors_accept_exact_integer_storage_and_columns_reject() {
        let row =
            Tensor::new_integer(IntegerStorage::U64(vec![0, u64::MAX, 1]), vec![1, 3]).unwrap();
        let Value::Tensor(value) =
            run_static(EmptyClass::Uint64, vec![Value::Tensor(row)]).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_eq!(value.shape, vec![0, usize::MAX]);

        let column = Tensor::new_integer(IntegerStorage::U8(vec![0, 3]), vec![2, 1]).unwrap();
        let err = run_static(EmptyClass::Uint8, vec![Value::Tensor(column)]).unwrap_err();
        assert!(err.to_string().contains("row vector"));
    }

    #[test]
    fn nonempty_shapes_reject_without_multiplication() {
        let err = run_static(
            EmptyClass::Uint64,
            vec![
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
        )
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:empty:NonEmptyShape"));
    }

    #[test]
    fn global_shorthand_is_gated_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let err = futures::executor::block_on(empty_builtin(vec![
            Value::Int(IntValue::U8(0)),
            Value::from("uint8"),
        ]))
        .unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:EmptyGlobalCallExtension")
        );
    }

    #[test]
    fn resident_size_control_rejects_before_provider_access_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let resident = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
            shape: vec![1, 2],
            device_id: u32::MAX,
            buffer_id: u64::MAX,
        });
        let err = futures::executor::block_on(parse_shape(&[resident])).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:EmptyResidentSizeExtension")
        );
    }

    #[test]
    fn integer_static_names_are_registered() {
        for name in [
            "int8.empty",
            "int16.empty",
            "int32.empty",
            "int64.empty",
            "uint8.empty",
            "uint16.empty",
            "uint32.empty",
            "uint64.empty",
        ] {
            let builtin = runmat_builtins::builtin_function_by_name(name)
                .unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(builtin.integer_capabilities.len(), 1);
        }
    }

    #[test]
    fn dotted_integer_static_dispatch_preserves_wide_shape_and_class() {
        let value = futures::executor::block_on(crate::call_builtin_async(
            "uint64.empty",
            &[
                Value::Int(IntValue::I64(-1)),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
        ))
        .unwrap();
        let Value::Tensor(value) = value else {
            panic!("expected uint64 tensor");
        };
        assert_eq!(value.shape, vec![0, usize::MAX]);
        assert_eq!(value.numeric_dtype(), NumericDType::U64);
        assert!(value.integer_storage().is_some());
    }
}
