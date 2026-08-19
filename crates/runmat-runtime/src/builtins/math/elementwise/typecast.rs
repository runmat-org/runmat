//! Byte-preserving numeric reinterpretation for MATLAB-compatible `typecast`.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ComplexStorage, ComplexTensor, IntValue, IntegerComplexStorage,
    IntegerStorage, LogicalArray, NumericStorage, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

const NAME: &str = "typecast";

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Reinterpreted scalar or vector.",
}];
const NEWTYPE_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full numeric or logical scalar or vector.",
    },
    BuiltinParamDescriptor {
        name: "newtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric or logical output class.",
    },
];
const LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full numeric or logical scalar or vector.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Literal string \"like\".",
    },
    BuiltinParamDescriptor {
        name: "prototype",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host output class and complexity prototype.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = typecast(X, newtype)",
        inputs: &NEWTYPE_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = typecast(X, \"like\", prototype)",
        inputs: &LIKE_INPUTS,
        outputs: &OUTPUT,
    },
];
const INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.INVALID_ARGUMENT",
    identifier: Some("RunMat:typecast:InvalidArgument"),
    when: "The requested class, prototype, or argument list is invalid.",
    message: "typecast: invalid argument",
};
const INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.INVALID_INPUT",
    identifier: Some("RunMat:typecast:InvalidInput"),
    when: "The input is sparse, is not a scalar or vector, or has an incompatible byte count.",
    message: "typecast: invalid input",
};
const GPU_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.GPU_UNSUPPORTED",
    identifier: Some("RunMat:typecast:GpuUnsupported"),
    when: "A GPU call uses complex or logical input or the like syntax.",
    message: "typecast: unsupported gpuArray form",
};
const INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TYPECAST.INTERNAL",
    identifier: Some("RunMat:typecast:Internal"),
    when: "Exact gather, reconstruction, or resident restoration fails.",
    message: "typecast: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 4] =
    [INVALID_ARGUMENT, INVALID_INPUT, GPU_UNSUPPORTED, INTERNAL];

pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every native integer class is reinterpreted from its authoritative byte representation without numeric conversion.",
}];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = typecast(integer_X, newtype)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Scalar and vector inputs preserve their native byte sequence and orientation; output class and element count follow the requested element width.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = typecast(gpuArray(integer_X), newtype)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Real numeric resident input is transferred exactly through its owning provider, reinterpreted without floating conversion, and restored to that provider; complex, logical, and like GPU forms reject before transfer.",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetClass {
    F64,
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Logical,
}

impl TargetClass {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let Value::String(name) = value else {
            return Err(error(&INVALID_ARGUMENT, "newtype must be a string scalar"));
        };
        match name.to_ascii_lowercase().as_str() {
            "double" => Ok(Self::F64),
            "single" => Ok(Self::F32),
            "int8" => Ok(Self::I8),
            "int16" => Ok(Self::I16),
            "int32" => Ok(Self::I32),
            "int64" => Ok(Self::I64),
            "uint8" => Ok(Self::U8),
            "uint16" => Ok(Self::U16),
            "uint32" => Ok(Self::U32),
            "uint64" => Ok(Self::U64),
            "logical" => Ok(Self::Logical),
            "char" => Err(error(
                &INVALID_ARGUMENT,
                "character reinterpretation is not implemented",
            )),
            _ => Err(error(&INVALID_ARGUMENT, "unsupported output class")),
        }
    }

    fn width(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Logical => 1,
        }
    }
}

#[runtime_builtin(
    name = "typecast",
    category = "math/elementwise",
    summary = "Reinterpret the bytes of a numeric scalar or vector as another class.",
    keywords = "typecast,reinterpret,bytes,integer,single,double,gpuArray",
    accel = "custom",
    descriptor(crate::builtins::math::elementwise::typecast::DESCRIPTOR),
    integer_capabilities(crate::builtins::math::elementwise::typecast::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::elementwise::typecast"
)]
async fn typecast_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if !(2..=3).contains(&args.len()) {
        return Err(error(&INVALID_ARGUMENT, "expected two or three inputs"));
    }
    let mut args = args.into_iter();
    let source = args.next().expect("source");
    let selector = args.next().expect("selector");
    let prototype = args.next();
    let source_handle = match &source {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };

    let (target, complex_output) = match prototype {
        Some(prototype) => {
            if !matches!(&selector, Value::String(keyword) if keyword.eq_ignore_ascii_case("like"))
            {
                return Err(error(
                    &INVALID_ARGUMENT,
                    "three-input syntax requires the literal string \"like\"",
                ));
            }
            if source_handle.is_some() || matches!(prototype, Value::GpuTensor(_)) {
                return Err(terminal_gpu_error(
                    "gpuArray input does not support the like syntax",
                ));
            }
            target_from_prototype(&prototype)?
        }
        None => (TargetClass::parse(&selector)?, false),
    };

    if let Some(handle) = source_handle.as_ref() {
        if runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            || runmat_accelerate_api::handle_is_logical(handle)
        {
            return Err(terminal_gpu_error(
                "complex and logical gpuArray inputs are not supported",
            ));
        }
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(|cause| error(&INTERNAL, cause.message()))?;
        let result = reinterpret_host(gathered, target, false)?;
        return gpu_helpers::restore_class_preserving_value(handle, result, NAME)
            .map_err(|cause| error(&INTERNAL, cause.message()));
    }

    reinterpret_host(source, target, complex_output)
}

fn target_from_prototype(value: &Value) -> BuiltinResult<(TargetClass, bool)> {
    match value {
        Value::Num(_) => Ok((TargetClass::F64, false)),
        Value::Complex(_, _) => Ok((TargetClass::F64, true)),
        Value::Int(value) => Ok((target_from_int(value), false)),
        Value::Bool(_) | Value::LogicalArray(_) => Ok((TargetClass::Logical, false)),
        Value::Tensor(tensor) => Ok((target_from_tensor(tensor)?, false)),
        Value::ComplexTensor(tensor) => Ok((
            match tensor.numeric_dtype() {
                runmat_builtins::NumericDType::F64 => TargetClass::F64,
                runmat_builtins::NumericDType::F32 => TargetClass::F32,
                runmat_builtins::NumericDType::I8 => TargetClass::I8,
                runmat_builtins::NumericDType::I16 => TargetClass::I16,
                runmat_builtins::NumericDType::I32 => TargetClass::I32,
                runmat_builtins::NumericDType::I64 => TargetClass::I64,
                runmat_builtins::NumericDType::U8 => TargetClass::U8,
                runmat_builtins::NumericDType::U16 => TargetClass::U16,
                runmat_builtins::NumericDType::U32 => TargetClass::U32,
                runmat_builtins::NumericDType::U64 => TargetClass::U64,
            },
            true,
        )),
        _ => Err(error(&INVALID_ARGUMENT, "unsupported like prototype")),
    }
}

fn target_from_tensor(tensor: &Tensor) -> BuiltinResult<TargetClass> {
    Ok(match tensor.numeric_dtype() {
        runmat_builtins::NumericDType::F64 => TargetClass::F64,
        runmat_builtins::NumericDType::F32 => TargetClass::F32,
        runmat_builtins::NumericDType::I8 => TargetClass::I8,
        runmat_builtins::NumericDType::I16 => TargetClass::I16,
        runmat_builtins::NumericDType::I32 => TargetClass::I32,
        runmat_builtins::NumericDType::I64 => TargetClass::I64,
        runmat_builtins::NumericDType::U8 => TargetClass::U8,
        runmat_builtins::NumericDType::U16 => TargetClass::U16,
        runmat_builtins::NumericDType::U32 => TargetClass::U32,
        runmat_builtins::NumericDType::U64 => TargetClass::U64,
    })
}

fn target_from_int(value: &IntValue) -> TargetClass {
    match value {
        IntValue::I8(_) => TargetClass::I8,
        IntValue::I16(_) => TargetClass::I16,
        IntValue::I32(_) => TargetClass::I32,
        IntValue::I64(_) => TargetClass::I64,
        IntValue::U8(_) => TargetClass::U8,
        IntValue::U16(_) => TargetClass::U16,
        IntValue::U32(_) => TargetClass::U32,
        IntValue::U64(_) => TargetClass::U64,
    }
}

fn reinterpret_host(source: Value, target: TargetClass, complex: bool) -> BuiltinResult<Value> {
    let shape = source_shape(&source)?;
    validate_vector_shape(&shape)?;
    let bytes = source_bytes(source)?;
    let bytes_per_output = target
        .width()
        .checked_mul(if complex { 2 } else { 1 })
        .ok_or_else(|| error(&INVALID_INPUT, "output element width overflow"))?;
    if bytes.len() % bytes_per_output != 0 {
        return Err(error(
            &INVALID_INPUT,
            "input byte count is not divisible by the requested output element width",
        ));
    }
    let output_len = bytes.len() / bytes_per_output;
    let output_shape = output_vector_shape(&shape, output_len);
    let storage = decode_storage(&bytes, target)?;
    if complex {
        let storage = pair_complex_storage(storage)?;
        ComplexTensor::from_complex_storage(storage, output_shape)
            .map(Value::ComplexTensor)
            .map_err(|cause| error(&INTERNAL, cause))
    } else if target == TargetClass::Logical {
        let NumericStorage::U8(values) = storage else {
            unreachable!("logical decoder emits u8 storage")
        };
        LogicalArray::new(values, output_shape)
            .map(Value::LogicalArray)
            .map_err(|cause| error(&INTERNAL, cause))
    } else {
        Tensor::from_numeric_storage(storage, output_shape)
            .map(Value::Tensor)
            .map_err(|cause| error(&INTERNAL, cause))
    }
}

fn source_shape(source: &Value) -> BuiltinResult<Vec<usize>> {
    match source {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(vec![1, 1]),
        Value::Tensor(tensor) => Ok(tensor.shape.clone()),
        Value::ComplexTensor(tensor) => Ok(tensor.shape.clone()),
        Value::LogicalArray(array) => Ok(array.shape.clone()),
        Value::SparseTensor(_) => Err(error(&INVALID_INPUT, "input must be full, not sparse")),
        _ => Err(error(&INVALID_INPUT, "input must be numeric or logical")),
    }
}

fn validate_vector_shape(shape: &[usize]) -> BuiltinResult<()> {
    if shape.iter().filter(|&&dimension| dimension > 1).count() > 1 {
        return Err(error(&INVALID_INPUT, "input must be a scalar or vector"));
    }
    Ok(())
}

fn output_vector_shape(input: &[usize], len: usize) -> Vec<usize> {
    if input.contains(&0) {
        if input.first() == Some(&1) {
            return vec![1, 0];
        }
        if input.get(1) == Some(&1) {
            return vec![0, 1];
        }
        return vec![0, 0];
    }
    if input.first() == Some(&1) || input.iter().all(|dimension| *dimension == 1) {
        vec![1, len]
    } else {
        vec![len, 1]
    }
}

fn source_bytes(source: Value) -> BuiltinResult<Vec<u8>> {
    match source {
        Value::Num(value) => Ok(value.to_ne_bytes().to_vec()),
        Value::Int(value) => Ok(int_bytes(value)),
        Value::Bool(value) => Ok(vec![u8::from(value)]),
        Value::Complex(real, imag) => {
            let mut bytes = real.to_ne_bytes().to_vec();
            bytes.extend_from_slice(&imag.to_ne_bytes());
            Ok(bytes)
        }
        Value::Tensor(tensor) => tensor
            .into_numeric_storage()
            .map(storage_bytes)
            .map_err(|cause| error(&INTERNAL, cause)),
        Value::LogicalArray(array) => Ok(array.data),
        Value::ComplexTensor(tensor) => Ok(complex_storage_bytes(tensor.into_complex_storage())),
        _ => Err(error(&INVALID_INPUT, "input must be numeric or logical")),
    }
}

fn int_bytes(value: IntValue) -> Vec<u8> {
    match value {
        IntValue::I8(value) => value.to_ne_bytes().to_vec(),
        IntValue::I16(value) => value.to_ne_bytes().to_vec(),
        IntValue::I32(value) => value.to_ne_bytes().to_vec(),
        IntValue::I64(value) => value.to_ne_bytes().to_vec(),
        IntValue::U8(value) => value.to_ne_bytes().to_vec(),
        IntValue::U16(value) => value.to_ne_bytes().to_vec(),
        IntValue::U32(value) => value.to_ne_bytes().to_vec(),
        IntValue::U64(value) => value.to_ne_bytes().to_vec(),
    }
}

fn storage_bytes(storage: NumericStorage) -> Vec<u8> {
    macro_rules! encode {
        ($values:expr, $ty:ty) => {{
            let values = $values;
            let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<$ty>());
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
            bytes
        }};
    }
    match storage {
        NumericStorage::F64(values) => encode!(values, f64),
        NumericStorage::F32(values) => encode!(values, f32),
        NumericStorage::I8(values) => values.into_iter().map(|value| value as u8).collect(),
        NumericStorage::I16(values) => encode!(values, i16),
        NumericStorage::I32(values) => encode!(values, i32),
        NumericStorage::I64(values) => encode!(values, i64),
        NumericStorage::U8(values) => values,
        NumericStorage::U16(values) => encode!(values, u16),
        NumericStorage::U32(values) => encode!(values, u32),
        NumericStorage::U64(values) => encode!(values, u64),
    }
}

fn complex_storage_bytes(storage: ComplexStorage) -> Vec<u8> {
    match storage {
        ComplexStorage::F64(values) => values
            .into_iter()
            .flat_map(|(real, imag)| real.to_ne_bytes().into_iter().chain(imag.to_ne_bytes()))
            .collect(),
        ComplexStorage::F32(values) => values
            .into_iter()
            .flat_map(|(real, imag)| real.to_ne_bytes().into_iter().chain(imag.to_ne_bytes()))
            .collect(),
        ComplexStorage::Integer(values) => {
            let len = values.len();
            let mut bytes = Vec::new();
            for index in 0..len {
                bytes.extend(int_bytes(
                    values
                        .real
                        .value_at(index)
                        .expect("validated real component"),
                ));
                bytes.extend(int_bytes(
                    values
                        .imag
                        .value_at(index)
                        .expect("validated imaginary component"),
                ));
            }
            bytes
        }
    }
}

fn decode_storage(bytes: &[u8], target: TargetClass) -> BuiltinResult<NumericStorage> {
    macro_rules! decode {
        ($ty:ty, $variant:ident) => {{
            let values = bytes
                .chunks_exact(std::mem::size_of::<$ty>())
                .map(|chunk| <$ty>::from_ne_bytes(chunk.try_into().expect("exact chunk")))
                .collect();
            NumericStorage::$variant(values)
        }};
    }
    Ok(match target {
        TargetClass::F64 => decode!(f64, F64),
        TargetClass::F32 => decode!(f32, F32),
        TargetClass::I8 => NumericStorage::I8(bytes.iter().map(|byte| *byte as i8).collect()),
        TargetClass::I16 => decode!(i16, I16),
        TargetClass::I32 => decode!(i32, I32),
        TargetClass::I64 => decode!(i64, I64),
        TargetClass::U8 => NumericStorage::U8(bytes.to_vec()),
        TargetClass::U16 => decode!(u16, U16),
        TargetClass::U32 => decode!(u32, U32),
        TargetClass::U64 => decode!(u64, U64),
        TargetClass::Logical => {
            NumericStorage::U8(bytes.iter().map(|byte| u8::from(*byte != 0)).collect())
        }
    })
}

macro_rules! integer_complex {
    ($values:expr, $variant:ident) => {{
        let mut real = Vec::with_capacity($values.len() / 2);
        let mut imag = Vec::with_capacity($values.len() / 2);
        let mut values = $values.into_iter();
        while let Some(value) = values.next() {
            real.push(value);
            imag.push(values.next().expect("complex byte count was validated"));
        }
        ComplexStorage::Integer(
            IntegerComplexStorage::new(
                IntegerStorage::$variant(real),
                IntegerStorage::$variant(imag),
            )
            .expect("paired typecast storage has matching class and length"),
        )
    }};
}

fn pair_complex_storage(storage: NumericStorage) -> BuiltinResult<ComplexStorage> {
    macro_rules! pairs {
        ($values:expr, $variant:ident) => {{
            let mut real = Vec::with_capacity($values.len() / 2);
            let mut imag = Vec::with_capacity($values.len() / 2);
            let mut values = $values.into_iter();
            while let Some(value) = values.next() {
                real.push(value);
                imag.push(values.next().expect("complex byte count was validated"));
            }
            (real, imag, stringify!($variant))
        }};
    }
    Ok(match storage {
        NumericStorage::F64(values) => {
            let (real, imag, _) = pairs!(values, F64);
            ComplexStorage::F64(real.into_iter().zip(imag).collect())
        }
        NumericStorage::F32(values) => {
            let (real, imag, _) = pairs!(values, F32);
            ComplexStorage::F32(real.into_iter().zip(imag).collect())
        }
        NumericStorage::I8(values) => integer_complex!(values, I8),
        NumericStorage::I16(values) => integer_complex!(values, I16),
        NumericStorage::I32(values) => integer_complex!(values, I32),
        NumericStorage::I64(values) => integer_complex!(values, I64),
        NumericStorage::U8(values) => integer_complex!(values, U8),
        NumericStorage::U16(values) => integer_complex!(values, U16),
        NumericStorage::U32(values) => integer_complex!(values, U32),
        NumericStorage::U64(values) => integer_complex!(values, U64),
    })
}

fn terminal_gpu_error(detail: impl std::fmt::Display) -> RuntimeError {
    build_runtime_error(format!("{}: {detail}", GPU_UNSUPPORTED.message))
        .with_builtin(NAME)
        .with_identifier(GPU_UNSUPPORTED.identifier.expect("gpu error identifier"))
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn error(
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", descriptor.message)).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::typecast_builtin;
    use futures::executor::block_on;
    use runmat_builtins::{ComplexTensor, IntValue, IntegerStorage, Tensor, Value};

    use crate::builtins::common::{gpu_helpers, test_support};

    #[test]
    fn integer_typecast_preserves_bytes_width_and_orientation() {
        let input =
            Tensor::new_integer(IntegerStorage::U32(vec![1, 255, 256]), vec![1, 3]).expect("input");
        let Value::Tensor(output) = block_on(typecast_builtin(vec![
            Value::Tensor(input),
            Value::String("uint8".to_string()),
        ]))
        .expect("typecast") else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![1, 12]);
        let mut expected = Vec::new();
        for value in [1_u32, 255, 256] {
            expected.extend_from_slice(&value.to_ne_bytes());
        }
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(expected))
        );
    }

    #[test]
    fn integer_typecast_round_trips_exact_uint64_bits() {
        let original = [0_u64, (1_u64 << 63) + 17, u64::MAX];
        let input =
            Tensor::new_integer(IntegerStorage::U64(original.to_vec()), vec![3, 1]).expect("input");
        let bytes = block_on(typecast_builtin(vec![
            Value::Tensor(input),
            Value::String("uint8".to_string()),
        ]))
        .expect("to bytes");
        let Value::Tensor(output) = block_on(typecast_builtin(vec![
            bytes,
            Value::String("uint64".to_string()),
        ]))
        .expect("from bytes") else {
            panic!("expected tensor");
        };
        assert_eq!(output.shape, vec![3, 1]);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(original.to_vec()))
        );
    }

    #[test]
    fn like_complex_integer_groups_adjacent_components_exactly() {
        let input = Tensor::new_integer(
            IntegerStorage::I16(vec![-1, 2, i16::MIN, i16::MAX]),
            vec![1, 4],
        )
        .expect("input");
        let prototype = ComplexTensor::new_integer(
            runmat_builtins::IntegerComplexStorage::new(
                IntegerStorage::I16(vec![0]),
                IntegerStorage::I16(vec![0]),
            )
            .expect("prototype storage"),
            vec![1, 1],
        )
        .expect("prototype");
        let Value::ComplexTensor(output) = block_on(typecast_builtin(vec![
            Value::Tensor(input),
            Value::String("like".to_string()),
            Value::ComplexTensor(prototype),
        ]))
        .expect("complex like") else {
            panic!("expected complex tensor");
        };
        assert_eq!(output.shape, vec![1, 2]);
        let storage = output.integer_storage().expect("integer complex");
        assert_eq!(storage.real, IntegerStorage::I16(vec![-1, i16::MIN]));
        assert_eq!(storage.imag, IntegerStorage::I16(vec![2, i16::MAX]));
    }

    #[test]
    fn typecast_rejects_matrix_and_indivisible_byte_count() {
        let matrix =
            Tensor::new_integer(IntegerStorage::U8(vec![1, 2, 3, 4]), vec![2, 2]).expect("matrix");
        assert!(block_on(typecast_builtin(vec![
            Value::Tensor(matrix),
            Value::String("uint16".to_string()),
        ]))
        .is_err());
        assert!(block_on(typecast_builtin(vec![
            Value::Int(IntValue::U8(1)),
            Value::String("uint16".to_string()),
        ]))
        .is_err());
    }

    #[test]
    fn typecast_is_registered_and_dispatches_exact_integer_storage() {
        let input = Tensor::new_integer(IntegerStorage::I16(vec![-1, i16::MIN]), vec![1, 2])
            .expect("input");
        let Value::Tensor(output) = crate::dispatcher::call_builtin(
            "typecast",
            &[Value::Tensor(input), Value::String("uint16".to_string())],
        )
        .expect("registered typecast") else {
            panic!("expected tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U16(vec![u16::MAX, 1_u16 << 15]))
        );
    }

    #[test]
    fn real_integer_gpu_typecast_preserves_owner_residency_and_exact_bytes() {
        test_support::with_test_provider(|provider| {
            let source = Tensor::new_integer(
                IntegerStorage::U64(vec![(1_u64 << 63) + 9, u64::MAX]),
                vec![1, 2],
            )
            .expect("source");
            let handle = gpu_helpers::upload_tensor(provider, &source).expect("upload");
            let output = block_on(typecast_builtin(vec![
                Value::GpuTensor(handle),
                Value::String("uint8".to_string()),
            ]))
            .expect("resident typecast");
            assert!(matches!(output, Value::GpuTensor(_)));
            let output = test_support::gather(output).expect("gather result");
            let mut expected = Vec::new();
            for value in [(1_u64 << 63) + 9, u64::MAX] {
                expected.extend_from_slice(&value.to_ne_bytes());
            }
            assert_eq!(
                output.integer_storage(),
                Some(&IntegerStorage::U8(expected))
            );
        });
    }
}
