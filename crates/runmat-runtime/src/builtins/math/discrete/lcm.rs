//! MATLAB-compatible `lcm` builtin.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, IntegerStorage, NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "lcm";
const GCD_BUILTIN_NAME: &str = "gcd";

const LCM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Least common multiples of A and B.",
}];

const LCM_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real positive integer scalar, vector, or array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real positive integer scalar, vector, or array.",
    },
];

const LCM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "L = lcm(A, B)",
    inputs: &LCM_INPUTS,
    outputs: &LCM_OUTPUT,
}];

const LCM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.INVALID_INPUT",
    identifier: Some("RunMat:lcm:InvalidInput"),
    when: "Inputs are not real positive integer numeric values, or integer class mixing is unsupported.",
    message: "lcm: invalid input",
};

const LCM_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.SIZE_MISMATCH",
    identifier: Some("RunMat:lcm:SizeMismatch"),
    when: "Inputs are neither the same size nor scalar-expandable.",
    message: "lcm: input sizes are not compatible",
};

const LCM_ERROR_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.OVERFLOW",
    identifier: Some("RunMat:lcm:Overflow"),
    when: "The least common multiple cannot be represented in the output numeric class.",
    message: "lcm: result overflows output type",
};

const LCM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LCM.INTERNAL",
    identifier: Some("RunMat:lcm:Internal"),
    when: "GPU gather or tensor construction fails.",
    message: "lcm: internal error",
};

const LCM_ERRORS: [BuiltinErrorDescriptor; 4] = [
    LCM_ERROR_INVALID_INPUT,
    LCM_ERROR_SIZE_MISMATCH,
    LCM_ERROR_OVERFLOW,
    LCM_ERROR_INTERNAL,
];

pub const LCM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LCM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LCM_ERRORS,
};

const GCD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "G",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Greatest common divisors of A and B.",
}];

const GCD_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real integer-valued scalar, vector, or array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real integer-valued scalar, vector, or array.",
    },
];

const GCD_EXTENDED_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "G",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Greatest common divisors of A and B.",
    },
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First Bézout coefficient satisfying A.*U + B.*V = G.",
    },
    BuiltinParamDescriptor {
        name: "V",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second Bézout coefficient satisfying A.*U + B.*V = G.",
    },
];

const GCD_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "G = gcd(A, B)",
        inputs: &GCD_INPUTS,
        outputs: &GCD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "[G, U, V] = gcd(A, B)",
        inputs: &GCD_INPUTS,
        outputs: &GCD_EXTENDED_OUTPUTS,
    },
];

const GCD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.INVALID_INPUT",
    identifier: Some("RunMat:gcd:InvalidInput"),
    when:
        "Inputs are not real integer-valued numeric values, or integer class mixing is unsupported.",
    message: "gcd: invalid input",
};

const GCD_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.SIZE_MISMATCH",
    identifier: Some("RunMat:gcd:SizeMismatch"),
    when: "Inputs are neither the same size nor scalar-expandable.",
    message: "gcd: input sizes are not compatible",
};

const GCD_ERROR_OVERFLOW: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.OVERFLOW",
    identifier: Some("RunMat:gcd:Overflow"),
    when: "The greatest common divisor or a requested Bézout coefficient cannot be represented in the output numeric class.",
    message: "gcd: output overflows numeric class",
};

const GCD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GCD.INTERNAL",
    identifier: Some("RunMat:gcd:Internal"),
    when: "GPU gather or tensor construction fails.",
    message: "gcd: internal error",
};

const GCD_ERRORS: [BuiltinErrorDescriptor; 4] = [
    GCD_ERROR_INVALID_INPUT,
    GCD_ERROR_SIZE_MISMATCH,
    GCD_ERROR_OVERFLOW,
    GCD_ERROR_INTERNAL,
];

pub const GCD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GCD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GCD_ERRORS,
};

const LCM_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "If A is integer, B must use the same integer class or be a scalar double.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "If B is integer, A must use the same integer class or be a scalar double.",
    },
];

pub const LCM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "L = lcm(A, B)",
        inputs: &LCM_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::EvidenceOpen,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Inputs must be positive integer-valued real data; the exact overflow outcome remains evidence-gated.",
    }];

const GCD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "If A is integer, B must use the same integer class or be a scalar double.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "If B is integer, A must use the same integer class or be a scalar double.",
    },
];

const GCD_EXTENDED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::SIGNED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Extended outputs support signed integer, single, and double inputs.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::SIGNED_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Extended outputs support signed integer, single, and double inputs.",
    },
];

pub const GCD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "G = gcd(A, B)",
        inputs: &GCD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Signed and zero values are accepted; G is always nonnegative.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "[G, U, V] = gcd(A, B)",
        inputs: &GCD_EXTENDED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveNondoubleInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::SameSizeOrScalar,
        notes: "Unsigned integer inputs are rejected because requested Bézout coefficients require a signed result class.",
    },
];

fn lcm_type(args: &[Type], _ctx: &runmat_builtins::ResolveContext) -> Type {
    if args.iter().all(|ty| matches!(ty, Type::Int)) {
        Type::Int
    } else if args.iter().all(|ty| matches!(ty, Type::Num | Type::Int)) {
        Type::Num
    } else {
        Type::tensor()
    }
}

#[runtime_builtin(
    name = "lcm",
    category = "math/discrete",
    summary = "Compute least common multiples for positive integer inputs.",
    keywords = "lcm,least common multiple,integer,number theory,discrete",
    accel = "gather",
    type_resolver(lcm_type),
    descriptor(crate::builtins::math::discrete::lcm::LCM_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::discrete::lcm::LCM_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::discrete::lcm"
)]
async fn lcm_builtin(left: Value, right: Value) -> BuiltinResult<Value> {
    evaluate_lcm(left, right, &LCM_CONTEXT).await
}

#[runtime_builtin(
    name = "gcd",
    category = "math/discrete",
    summary = "Compute greatest common divisors for real integer-valued inputs.",
    keywords = "gcd,greatest common divisor,integer,number theory,discrete",
    accel = "gather",
    type_resolver(lcm_type),
    descriptor(crate::builtins::math::discrete::lcm::GCD_DESCRIPTOR),
    integer_capabilities(crate::builtins::math::discrete::lcm::GCD_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::math::discrete::lcm"
)]
async fn gcd_builtin(left: Value, right: Value) -> BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if matches!(requested, Some(count) if count > 3) {
        return Err(error_with_detail(
            &GCD_CONTEXT,
            GCD_CONTEXT.invalid,
            "at most three outputs are supported",
        ));
    }
    if requested == Some(0) {
        return Ok(Value::OutputList(Vec::new()));
    }
    let left = LcmInput::from_value(left, &GCD_CONTEXT).await?;
    let right = LcmInput::from_value(right, &GCD_CONTEXT).await?;
    let output = resolve_output_kind(&left, &right, &GCD_CONTEXT)?;
    let plan = SameSizeOrScalarPlan::new(&left, &right, &GCD_CONTEXT)?;
    let mut gcds = Vec::with_capacity(plan.len());
    let extended = requested.is_some_and(|count| count >= 2);
    let mut first_coefficients = extended.then(|| Vec::with_capacity(plan.len()));
    let mut second_coefficients = extended.then(|| Vec::with_capacity(plan.len()));
    if extended
        && matches!(
            output.class,
            NumericClass::U8 | NumericClass::U16 | NumericClass::U32 | NumericClass::U64
        )
    {
        return Err(error_with_detail(
            &GCD_CONTEXT,
            GCD_CONTEXT.invalid,
            "Bézout coefficients require double, single, or signed integer inputs",
        ));
    }
    for (left_idx, right_idx) in plan.iter() {
        if extended {
            let (gcd, mut first, mut second) =
                extended_gcd_u128(left.data[left_idx], right.data[right_idx]);
            if left.negative[left_idx] {
                first = -first;
            }
            if right.negative[right_idx] {
                second = -second;
            }
            gcds.push(gcd);
            first_coefficients
                .as_mut()
                .expect("extended output")
                .push(first);
            second_coefficients
                .as_mut()
                .expect("extended output")
                .push(second);
        } else {
            gcds.push(gcd_u128(left.data[left_idx], right.data[right_idx]));
        }
    }
    let gcd = value_from_lcms(gcds, plan.output_shape.clone(), output, &GCD_CONTEXT)?;
    let Some(count) = requested else {
        return Ok(gcd);
    };
    if count == 1 {
        return Ok(Value::OutputList(vec![gcd]));
    }
    let first = value_from_coefficients(
        first_coefficients.expect("extended output"),
        plan.output_shape.clone(),
        output,
    )?;
    if count == 2 {
        return Ok(Value::OutputList(vec![gcd, first]));
    }
    let second = value_from_coefficients(
        second_coefficients.expect("extended output"),
        plan.output_shape,
        output,
    )?;
    Ok(Value::OutputList(vec![gcd, first, second]))
}

struct BinaryContext {
    name: &'static str,
    invalid: &'static BuiltinErrorDescriptor,
    size_mismatch: &'static BuiltinErrorDescriptor,
    overflow: &'static BuiltinErrorDescriptor,
    internal: &'static BuiltinErrorDescriptor,
    accepts_zero_or_negative: bool,
}

const LCM_CONTEXT: BinaryContext = BinaryContext {
    name: BUILTIN_NAME,
    invalid: &LCM_ERROR_INVALID_INPUT,
    size_mismatch: &LCM_ERROR_SIZE_MISMATCH,
    overflow: &LCM_ERROR_OVERFLOW,
    internal: &LCM_ERROR_INTERNAL,
    accepts_zero_or_negative: false,
};

const GCD_CONTEXT: BinaryContext = BinaryContext {
    name: GCD_BUILTIN_NAME,
    invalid: &GCD_ERROR_INVALID_INPUT,
    size_mismatch: &GCD_ERROR_SIZE_MISMATCH,
    overflow: &GCD_ERROR_OVERFLOW,
    internal: &GCD_ERROR_INTERNAL,
    accepts_zero_or_negative: true,
};

async fn evaluate_lcm(
    left: Value,
    right: Value,
    context: &'static BinaryContext,
) -> BuiltinResult<Value> {
    let left = LcmInput::from_value(left, context).await?;
    let right = LcmInput::from_value(right, context).await?;
    let output_kind = resolve_output_kind(&left, &right, context)?;
    let plan = SameSizeOrScalarPlan::new(&left, &right, context)?;
    let mut out = Vec::with_capacity(plan.len());
    for (left_idx, right_idx) in plan.iter() {
        out.push(lcm_u128(left.data[left_idx], right.data[right_idx]));
    }
    value_from_lcms(out, plan.output_shape, output_kind, context)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericClass {
    Double,
    Single,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl NumericClass {
    fn is_float(self) -> bool {
        matches!(self, Self::Double | Self::Single)
    }

    fn is_integer(self) -> bool {
        !self.is_float()
    }

    fn max_value(self) -> u128 {
        match self {
            Self::Double | Self::Single => u128::MAX,
            Self::I8 => i8::MAX as u128,
            Self::I16 => i16::MAX as u128,
            Self::I32 => i32::MAX as u128,
            Self::I64 => i64::MAX as u128,
            Self::U8 => u8::MAX as u128,
            Self::U16 => u16::MAX as u128,
            Self::U32 => u32::MAX as u128,
            Self::U64 => u64::MAX as u128,
        }
    }

    fn signed_bounds(self) -> Option<(i128, i128)> {
        match self {
            Self::I8 => Some((i128::from(i8::MIN), i128::from(i8::MAX))),
            Self::I16 => Some((i128::from(i16::MIN), i128::from(i16::MAX))),
            Self::I32 => Some((i128::from(i32::MIN), i128::from(i32::MAX))),
            Self::I64 => Some((i128::from(i64::MIN), i128::from(i64::MAX))),
            Self::Double | Self::Single => None,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => {
                unreachable!("unsigned classes do not support Bézout coefficients")
            }
        }
    }

    fn tensor_dtype(self) -> Option<NumericDType> {
        match self {
            Self::Double => Some(NumericDType::F64),
            Self::Single => Some(NumericDType::F32),
            Self::U8 => Some(NumericDType::U8),
            Self::U16 => Some(NumericDType::U16),
            Self::U32 => Some(NumericDType::U32),
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::U64 => None,
        }
    }
}

struct LcmInput {
    data: Vec<u128>,
    negative: Vec<bool>,
    shape: Vec<usize>,
    class: NumericClass,
    native_integer_storage: bool,
}

#[derive(Clone, Copy)]
struct LcmOutput {
    class: NumericClass,
    native_integer_storage: bool,
}

impl LcmInput {
    fn is_scalar(&self) -> bool {
        self.data.len() == 1 && element_count(&self.shape) == 1
    }

    async fn from_value(value: Value, context: &'static BinaryContext) -> BuiltinResult<Self> {
        match value {
            Value::Num(value) => Ok(Self {
                data: vec![integer_magnitude_from_f64(value, context)?],
                negative: vec![value < 0.0],
                shape: vec![1, 1],
                class: NumericClass::Double,
                native_integer_storage: false,
            }),
            Value::Int(value) => Self::from_int_value(value, context),
            Value::Tensor(tensor) => Self::from_tensor(tensor, context),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|err| error_with_detail(context, context.internal, err))?;
                Self::from_tensor(tensor, context)
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => Err(error_with_detail(
                context,
                context.invalid,
                "inputs must be real",
            )),
            Value::Bool(_) | Value::LogicalArray(_) => Err(error_with_detail(
                context,
                context.invalid,
                "logical inputs are not numeric integer classes",
            )),
            other => Err(error_with_detail(
                context,
                context.invalid,
                format!("unsupported input type {other:?}"),
            )),
        }
    }

    fn from_int_value(value: IntValue, context: &'static BinaryContext) -> BuiltinResult<Self> {
        let negative = int_value_is_negative(&value);
        let (data, class) = match value {
            IntValue::I8(value) => (
                integer_magnitude_from_i128(i128::from(value), context)?,
                NumericClass::I8,
            ),
            IntValue::I16(value) => (
                integer_magnitude_from_i128(i128::from(value), context)?,
                NumericClass::I16,
            ),
            IntValue::I32(value) => (
                integer_magnitude_from_i128(i128::from(value), context)?,
                NumericClass::I32,
            ),
            IntValue::I64(value) => (
                integer_magnitude_from_i128(i128::from(value), context)?,
                NumericClass::I64,
            ),
            IntValue::U8(value) => (
                integer_magnitude_from_u128(u128::from(value), context)?,
                NumericClass::U8,
            ),
            IntValue::U16(value) => (
                integer_magnitude_from_u128(u128::from(value), context)?,
                NumericClass::U16,
            ),
            IntValue::U32(value) => (
                integer_magnitude_from_u128(u128::from(value), context)?,
                NumericClass::U32,
            ),
            IntValue::U64(value) => (
                integer_magnitude_from_u128(u128::from(value), context)?,
                NumericClass::U64,
            ),
        };
        Ok(Self {
            data: vec![data],
            negative: vec![negative],
            shape: vec![1, 1],
            class,
            native_integer_storage: true,
        })
    }

    fn from_tensor(tensor: Tensor, context: &'static BinaryContext) -> BuiltinResult<Self> {
        let shape = tensor.shape.clone();
        let storage = tensor
            .into_numeric_storage()
            .map_err(|err| error_with_detail(context, context.internal, err))?;
        let negative = match &storage {
            NumericStorage::F64(values) => values.iter().map(|&value| value < 0.0).collect(),
            NumericStorage::F32(values) => values.iter().map(|&value| value < 0.0).collect(),
            NumericStorage::I8(values) => values.iter().map(|&value| value < 0).collect(),
            NumericStorage::I16(values) => values.iter().map(|&value| value < 0).collect(),
            NumericStorage::I32(values) => values.iter().map(|&value| value < 0).collect(),
            NumericStorage::I64(values) => values.iter().map(|&value| value < 0).collect(),
            NumericStorage::U8(values) => vec![false; values.len()],
            NumericStorage::U16(values) => vec![false; values.len()],
            NumericStorage::U32(values) => vec![false; values.len()],
            NumericStorage::U64(values) => vec![false; values.len()],
        };
        let (data, class, native_integer_storage) = match storage {
            NumericStorage::F64(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_f64(value, context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::Double,
                false,
            ),
            NumericStorage::F32(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_f64(f64::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::Single,
                false,
            ),
            NumericStorage::I8(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_i128(i128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::I8,
                true,
            ),
            NumericStorage::I16(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_i128(i128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::I16,
                true,
            ),
            NumericStorage::I32(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_i128(i128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::I32,
                true,
            ),
            NumericStorage::I64(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_i128(i128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::I64,
                true,
            ),
            NumericStorage::U8(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_u128(u128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::U8,
                true,
            ),
            NumericStorage::U16(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_u128(u128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::U16,
                true,
            ),
            NumericStorage::U32(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_u128(u128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::U32,
                true,
            ),
            NumericStorage::U64(values) => (
                values
                    .into_iter()
                    .map(|value| integer_magnitude_from_u128(u128::from(value), context))
                    .collect::<BuiltinResult<Vec<_>>>()?,
                NumericClass::U64,
                true,
            ),
        };
        Ok(Self {
            data,
            negative,
            shape,
            class,
            native_integer_storage,
        })
    }
}

struct SameSizeOrScalarPlan {
    output_shape: Vec<usize>,
    len: usize,
    left_scalar: bool,
    right_scalar: bool,
}

impl SameSizeOrScalarPlan {
    fn new(
        left: &LcmInput,
        right: &LcmInput,
        context: &'static BinaryContext,
    ) -> BuiltinResult<Self> {
        let left_scalar = left.is_scalar();
        let right_scalar = right.is_scalar();
        let output_shape = if left.shape == right.shape {
            left.shape.clone()
        } else if left_scalar {
            right.shape.clone()
        } else if right_scalar {
            left.shape.clone()
        } else {
            return Err(error_with_detail(
                context,
                context.size_mismatch,
                "inputs must be the same size or one input must be scalar",
            ));
        };
        Ok(Self {
            len: element_count(&output_shape),
            output_shape,
            left_scalar,
            right_scalar,
        })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.len).map(|idx| {
            (
                if self.left_scalar { 0 } else { idx },
                if self.right_scalar { 0 } else { idx },
            )
        })
    }
}

fn resolve_output_kind(
    left: &LcmInput,
    right: &LcmInput,
    context: &'static BinaryContext,
) -> BuiltinResult<LcmOutput> {
    let (class, native_integer_storage) = match (left.class, right.class) {
        (a, b) if a == b => (
            a,
            left.native_integer_storage || right.native_integer_storage,
        ),
        (NumericClass::Double, NumericClass::Single)
        | (NumericClass::Single, NumericClass::Double) => (NumericClass::Single, false),
        (integer, NumericClass::Double) if integer.is_integer() && right.is_scalar() => {
            (integer, left.native_integer_storage)
        }
        (NumericClass::Double, integer) if integer.is_integer() && left.is_scalar() => {
            (integer, right.native_integer_storage)
        }
        (a, b) if a.is_integer() && b.is_integer() => {
            return Err(error_with_detail(
                context,
                context.invalid,
                "integer inputs must have the same class",
            ));
        }
        _ => {
            return Err(error_with_detail(
                context,
                context.invalid,
                "integer inputs can only be paired with the same class or a double scalar",
            ));
        }
    };
    Ok(LcmOutput {
        class,
        native_integer_storage,
    })
}

fn value_from_lcms(
    data: Vec<u128>,
    shape: Vec<usize>,
    output: LcmOutput,
    context: &'static BinaryContext,
) -> BuiltinResult<Value> {
    let class = output.class;
    for &value in &data {
        if value > class.max_value() {
            return Err(error_with_detail(
                context,
                context.overflow,
                "result exceeds output class range",
            ));
        }
    }

    if data.len() == 1 && element_count(&shape) == 1 {
        let value = data[0];
        return match class {
            NumericClass::Double => Ok(Value::Num(value as f64)),
            NumericClass::Single => Tensor::from_f32(vec![value as f32], shape)
                .map(Value::Tensor)
                .map_err(|err| error_with_detail(context, context.internal, err)),
            NumericClass::I8 => Ok(Value::Int(IntValue::I8(value as i8))),
            NumericClass::I16 => Ok(Value::Int(IntValue::I16(value as i16))),
            NumericClass::I32 => Ok(Value::Int(IntValue::I32(value as i32))),
            NumericClass::I64 => Ok(Value::Int(IntValue::I64(value as i64))),
            NumericClass::U8 => Ok(Value::Int(IntValue::U8(value as u8))),
            NumericClass::U16 => Ok(Value::Int(IntValue::U16(value as u16))),
            NumericClass::U32 => Ok(Value::Int(IntValue::U32(value as u32))),
            NumericClass::U64 => Ok(Value::Int(IntValue::U64(value as u64))),
        };
    }

    if output.native_integer_storage {
        return Tensor::new_integer(integer_storage_from_lcms(data, class), shape)
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(context, context.internal, err));
    }
    let dtype = class
        .tensor_dtype()
        .expect("floating LCM output classes have a tensor dtype");
    Tensor::new_with_dtype(
        data.into_iter()
            .map(|value| match class {
                NumericClass::Single => (value as f32) as f64,
                _ => value as f64,
            })
            .collect(),
        shape,
        dtype,
    )
    .map(Value::Tensor)
    .map_err(|err| error_with_detail(context, context.internal, err))
}

fn integer_storage_from_lcms(data: Vec<u128>, class: NumericClass) -> IntegerStorage {
    match class {
        NumericClass::I8 => IntegerStorage::I8(data.into_iter().map(|value| value as i8).collect()),
        NumericClass::I16 => {
            IntegerStorage::I16(data.into_iter().map(|value| value as i16).collect())
        }
        NumericClass::I32 => {
            IntegerStorage::I32(data.into_iter().map(|value| value as i32).collect())
        }
        NumericClass::I64 => {
            IntegerStorage::I64(data.into_iter().map(|value| value as i64).collect())
        }
        NumericClass::U8 => IntegerStorage::U8(data.into_iter().map(|value| value as u8).collect()),
        NumericClass::U16 => {
            IntegerStorage::U16(data.into_iter().map(|value| value as u16).collect())
        }
        NumericClass::U32 => {
            IntegerStorage::U32(data.into_iter().map(|value| value as u32).collect())
        }
        NumericClass::U64 => {
            IntegerStorage::U64(data.into_iter().map(|value| value as u64).collect())
        }
        NumericClass::Double | NumericClass::Single => {
            unreachable!("integer storage requires an integer output class")
        }
    }
}

fn value_from_coefficients(
    data: Vec<i128>,
    shape: Vec<usize>,
    output: LcmOutput,
) -> BuiltinResult<Value> {
    if let Some((min, max)) = output.class.signed_bounds() {
        if data.iter().any(|&value| value < min || value > max) {
            return Err(error_with_detail(
                &GCD_CONTEXT,
                GCD_CONTEXT.overflow,
                "Bézout coefficient exceeds output class range",
            ));
        }
    }
    if data.len() == 1 && element_count(&shape) == 1 {
        let value = data[0];
        return Ok(match output.class {
            NumericClass::Double => Value::Num(value as f64),
            NumericClass::Single => {
                return Tensor::from_f32(vec![value as f32], shape)
                    .map(Value::Tensor)
                    .map_err(|err| error_with_detail(&GCD_CONTEXT, GCD_CONTEXT.internal, err));
            }
            NumericClass::I8 => Value::Int(IntValue::I8(value as i8)),
            NumericClass::I16 => Value::Int(IntValue::I16(value as i16)),
            NumericClass::I32 => Value::Int(IntValue::I32(value as i32)),
            NumericClass::I64 => Value::Int(IntValue::I64(value as i64)),
            NumericClass::U8 | NumericClass::U16 | NumericClass::U32 | NumericClass::U64 => {
                unreachable!("unsigned classes do not support Bézout coefficients")
            }
        });
    }
    if output.native_integer_storage {
        let storage = match output.class {
            NumericClass::I8 => IntegerStorage::I8(data.into_iter().map(|v| v as i8).collect()),
            NumericClass::I16 => IntegerStorage::I16(data.into_iter().map(|v| v as i16).collect()),
            NumericClass::I32 => IntegerStorage::I32(data.into_iter().map(|v| v as i32).collect()),
            NumericClass::I64 => IntegerStorage::I64(data.into_iter().map(|v| v as i64).collect()),
            _ => unreachable!("native coefficient output requires a signed integer class"),
        };
        return Tensor::new_integer(storage, shape)
            .map(Value::Tensor)
            .map_err(|err| error_with_detail(&GCD_CONTEXT, GCD_CONTEXT.internal, err));
    }
    let dtype = match output.class {
        NumericClass::Double => NumericDType::F64,
        NumericClass::Single => NumericDType::F32,
        _ => unreachable!("floating coefficient output requires a floating class"),
    };
    Tensor::new_with_dtype(
        data.into_iter()
            .map(|value| match output.class {
                NumericClass::Single => (value as f32) as f64,
                _ => value as f64,
            })
            .collect(),
        shape,
        dtype,
    )
    .map(Value::Tensor)
    .map_err(|err| error_with_detail(&GCD_CONTEXT, GCD_CONTEXT.internal, err))
}

fn int_value_is_negative(value: &IntValue) -> bool {
    match value {
        IntValue::I8(value) => *value < 0,
        IntValue::I16(value) => *value < 0,
        IntValue::I32(value) => *value < 0,
        IntValue::I64(value) => *value < 0,
        IntValue::U8(_) | IntValue::U16(_) | IntValue::U32(_) | IntValue::U64(_) => false,
    }
}

fn integer_magnitude_from_i128(
    value: i128,
    context: &'static BinaryContext,
) -> BuiltinResult<u128> {
    if !context.accepts_zero_or_negative && value <= 0 {
        return Err(error_with_detail(
            context,
            context.invalid,
            "inputs must be positive integers",
        ));
    }
    Ok(value.unsigned_abs())
}

fn integer_magnitude_from_u128(
    value: u128,
    context: &'static BinaryContext,
) -> BuiltinResult<u128> {
    if !context.accepts_zero_or_negative && value == 0 {
        return Err(error_with_detail(
            context,
            context.invalid,
            "inputs must be positive integers",
        ));
    }
    Ok(value)
}

fn integer_magnitude_from_f64(value: f64, context: &'static BinaryContext) -> BuiltinResult<u128> {
    if !value.is_finite()
        || value.fract() != 0.0
        || (!context.accepts_zero_or_negative && value <= 0.0)
    {
        return Err(error_with_detail(
            context,
            context.invalid,
            if context.accepts_zero_or_negative {
                "inputs must be finite integer values"
            } else {
                "inputs must be finite positive integers"
            },
        ));
    }
    if value.abs() >= u64::MAX as f64 {
        return Err(error_with_detail(
            context,
            context.invalid,
            "input is too large",
        ));
    }
    Ok(value.abs() as u128)
}

fn lcm_u128(left: u128, right: u128) -> u128 {
    if left == 0 || right == 0 {
        return 0;
    }
    left / gcd_u128(left, right) * right
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let rem = left % right;
        left = right;
        right = rem;
    }
    left
}

fn extended_gcd_u128(left: u128, right: u128) -> (u128, i128, i128) {
    let (mut old_remainder, mut remainder) = (left, right);
    let (mut old_left, mut left_coefficient) = (1i128, 0i128);
    let (mut old_right, mut right_coefficient) = (0i128, 1i128);
    while remainder != 0 {
        let quotient = (old_remainder / remainder) as i128;
        (old_remainder, remainder) = (remainder, old_remainder - (quotient as u128) * remainder);
        (old_left, left_coefficient) = (left_coefficient, old_left - quotient * left_coefficient);
        (old_right, right_coefficient) =
            (right_coefficient, old_right - quotient * right_coefficient);
    }
    (old_remainder, old_left, old_right)
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn error_with_detail(
    context: &'static BinaryContext,
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    let mut builder =
        build_runtime_error(format!("{}: {detail}", error.message)).with_builtin(context.name);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn lcm_double_array_and_scalar() {
        let input = Tensor::new(vec![5.0, 17.0, 10.0, 60.0], vec![2, 2]).unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(input), Value::Num(45.0))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![2, 2]);
                assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                assert_eq!(tensor.materialize_f64(), vec![45.0, 765.0, 90.0, 180.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_preserves_unsigned_integer_class() {
        let left = Tensor::new_with_dtype(vec![255.0, 511.0, 15.0], vec![1, 3], NumericDType::U16)
            .unwrap();
        let right =
            Tensor::new_with_dtype(vec![15.0, 127.0, 1023.0], vec![1, 3], NumericDType::U16)
                .unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.numeric_dtype(), NumericDType::U16);
                assert_eq!(tensor.materialize_f64(), vec![255.0, 64897.0, 5115.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_integer_array_accepts_double_scalar_and_keeps_integer_class() {
        let left =
            Tensor::new_with_dtype(vec![6.0, 10.0, 21.0], vec![1, 3], NumericDType::U32).unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(left), Value::Num(15.0))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.numeric_dtype(), NumericDType::U32);
                assert_eq!(tensor.materialize_f64(), vec![30.0, 30.0, 105.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_preserves_all_native_integer_tensor_classes() {
        let cases = [
            (
                IntegerStorage::I8(vec![6, 10]),
                IntegerStorage::I8(vec![30, 30]),
            ),
            (
                IntegerStorage::I16(vec![6, 10]),
                IntegerStorage::I16(vec![30, 30]),
            ),
            (
                IntegerStorage::I32(vec![6, 10]),
                IntegerStorage::I32(vec![30, 30]),
            ),
            (
                IntegerStorage::I64(vec![6, 10]),
                IntegerStorage::I64(vec![30, 30]),
            ),
            (
                IntegerStorage::U8(vec![6, 10]),
                IntegerStorage::U8(vec![30, 30]),
            ),
            (
                IntegerStorage::U16(vec![6, 10]),
                IntegerStorage::U16(vec![30, 30]),
            ),
            (
                IntegerStorage::U32(vec![6, 10]),
                IntegerStorage::U32(vec![30, 30]),
            ),
            (
                IntegerStorage::U64(vec![6, 10]),
                IntegerStorage::U64(vec![30, 30]),
            ),
        ];

        for (input, expected) in cases {
            let input = Tensor::new_integer(input, vec![1, 2]).expect("native integer tensor");
            let out = block_on(lcm_builtin(Value::Tensor(input), Value::Num(15.0)))
                .expect("integer tensor with double scalar");
            match out {
                Value::Tensor(tensor) => {
                    assert_eq!(tensor.shape, vec![1, 2]);
                    assert_eq!(tensor.integer_storage(), Some(&expected));
                }
                other => panic!("expected tensor, got {other:?}"),
            }
        }
    }

    #[test]
    fn lcm_uses_exact_native_uint64_tensor_values() {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_995]),
            vec![1, 2],
        )
        .expect("native uint64 tensor");
        let out = block_on(lcm_builtin(Value::Tensor(input), Value::Num(1.0)))
            .expect("lcm with identity");
        match out {
            Value::Tensor(tensor) => assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U64(vec![
                    9_007_199_254_740_993,
                    9_007_199_254_740_995,
                ]))
            ),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn lcm_scalar_integer_output_preserves_width() {
        let out = block_on(lcm_builtin(
            Value::Int(IntValue::U8(12)),
            Value::Int(IntValue::U8(18)),
        ))
        .expect("lcm");
        assert_eq!(out, Value::Int(IntValue::U8(36)));
    }

    #[test]
    fn lcm_rejects_zero_negative_fractional_and_complex() {
        for value in [
            Value::Num(0.0),
            Value::Num(-2.0),
            Value::Num(2.5),
            Value::Complex(2.0, 0.0),
        ] {
            let err = block_on(lcm_builtin(value, Value::Num(3.0))).expect_err("invalid input");
            assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
        }
    }

    #[test]
    fn lcm_rejects_unrepresentable_double_u64_boundary_before_casting() {
        let err = block_on(lcm_builtin(Value::Num(u64::MAX as f64), Value::Num(3.0)))
            .expect_err("unrepresentable u64 boundary should fail before cast");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
        assert!(err.to_string().contains("input is too large"));

        let tensor = Tensor::new(vec![3.0, u64::MAX as f64], vec![1, 2]).unwrap();
        let err = block_on(lcm_builtin(Value::Tensor(tensor), Value::Num(3.0)))
            .expect_err("unrepresentable tensor entry should fail before cast");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
        assert!(err.to_string().contains("input is too large"));
    }

    #[test]
    fn lcm_rejects_mismatched_shapes_and_integer_classes() {
        let left = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let right = Tensor::new(vec![5.0, 7.0, 11.0], vec![1, 3]).unwrap();
        let err = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right)))
            .expect_err("shape mismatch");
        assert_eq!(err.identifier(), LCM_ERROR_SIZE_MISMATCH.identifier);

        let left = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let right = Tensor::new(vec![5.0, 7.0, 11.0], vec![1, 3]).unwrap();
        let err = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right)))
            .expect_err("implicit expansion is not supported for lcm");
        assert_eq!(err.identifier(), LCM_ERROR_SIZE_MISMATCH.identifier);

        let err = block_on(lcm_builtin(
            Value::Int(IntValue::U8(2)),
            Value::Int(IntValue::U16(4)),
        ))
        .expect_err("integer class mismatch");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);

        let single_scalar =
            Tensor::new_with_dtype(vec![3.0], vec![1, 1], NumericDType::F32).unwrap();
        let err = block_on(lcm_builtin(
            Value::Int(IntValue::U16(2)),
            Value::Tensor(single_scalar),
        ))
        .expect_err("integer plus single scalar is not permitted");
        assert_eq!(err.identifier(), LCM_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn lcm_rejects_integer_overflow() {
        let err = block_on(lcm_builtin(
            Value::Int(IntValue::U8(200)),
            Value::Int(IntValue::U8(201)),
        ))
        .expect_err("overflow");
        assert_eq!(err.identifier(), LCM_ERROR_OVERFLOW.identifier);
    }

    #[test]
    fn lcm_single_output_rounds_through_single_precision() {
        let left =
            Tensor::new_with_dtype(vec![16_777_217.0, 3.0], vec![1, 2], NumericDType::F32).unwrap();
        let right = Tensor::new_with_dtype(vec![1.0, 5.0], vec![1, 2], NumericDType::F32).unwrap();
        let out = block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right))).expect("lcm");
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
                assert_eq!(
                    tensor.materialize_f64(),
                    vec![(16_777_217_u128 as f32) as f64, 15.0]
                );
            }
            other => panic!("expected tensor, got {other:?}"),
        }

        let left = Tensor::from_f32(vec![6.0], vec![1, 1]).unwrap();
        let right = Tensor::from_f32(vec![15.0], vec![1, 1]).unwrap();
        let Value::Tensor(out) =
            block_on(lcm_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
        else {
            panic!("expected native-single scalar tensor")
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.as_f32_slice(), Some([30.0].as_slice()));
    }

    #[test]
    fn gcd_is_registered_and_handles_negative_and_zero_values() {
        assert!(runmat_builtins::builtin_function_by_name("gcd").is_some());
        let left = Tensor::new(vec![-5.0, 17.0, 10.0, 0.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![-15.0, 3.0, 100.0, 0.0], vec![2, 2]).unwrap();
        let Value::Tensor(out) =
            block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.numeric_dtype(), NumericDType::F64);
        assert_eq!(out.materialize_f64(), vec![5.0, 1.0, 10.0, 0.0]);
    }

    #[test]
    fn gcd_preserves_all_native_integer_classes_with_scalar_double() {
        let cases = [
            (
                IntegerStorage::I8(vec![-12, 18]),
                IntegerStorage::I8(vec![6, 6]),
            ),
            (
                IntegerStorage::I16(vec![-12, 18]),
                IntegerStorage::I16(vec![6, 6]),
            ),
            (
                IntegerStorage::I32(vec![-12, 18]),
                IntegerStorage::I32(vec![6, 6]),
            ),
            (
                IntegerStorage::I64(vec![-12, 18]),
                IntegerStorage::I64(vec![6, 6]),
            ),
            (
                IntegerStorage::U8(vec![12, 18]),
                IntegerStorage::U8(vec![6, 6]),
            ),
            (
                IntegerStorage::U16(vec![12, 18]),
                IntegerStorage::U16(vec![6, 6]),
            ),
            (
                IntegerStorage::U32(vec![12, 18]),
                IntegerStorage::U32(vec![6, 6]),
            ),
            (
                IntegerStorage::U64(vec![12, 18]),
                IntegerStorage::U64(vec![6, 6]),
            ),
        ];
        for (input, expected) in cases {
            let input = Tensor::new_integer(input, vec![1, 2]).unwrap();
            let Value::Tensor(out) =
                block_on(gcd_builtin(Value::Tensor(input), Value::Num(6.0))).unwrap()
            else {
                panic!("expected tensor");
            };
            assert_eq!(out.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn gcd_preserves_single_and_rejects_invalid_class_domain_and_shape() {
        let left = Tensor::from_f32(vec![12.0, 18.0], vec![1, 2]).unwrap();
        let right = Tensor::new(vec![8.0, 30.0], vec![1, 2]).unwrap();
        let Value::Tensor(out) =
            block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
        else {
            panic!("expected tensor");
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.as_f32_slice(), Some([4.0, 6.0].as_slice()));

        let left = Tensor::from_f32(vec![-12.0], vec![1, 1]).unwrap();
        let right = Tensor::from_f32(vec![18.0], vec![1, 1]).unwrap();
        let Value::Tensor(out) =
            block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
        else {
            panic!("expected native-single scalar tensor")
        };
        assert_eq!(out.numeric_dtype(), NumericDType::F32);
        assert_eq!(out.as_f32_slice(), Some([6.0].as_slice()));

        for (left, right) in [
            (Value::Num(2.5), Value::Num(1.0)),
            (Value::Complex(2.0, 0.0), Value::Num(1.0)),
            (Value::Int(IntValue::U8(2)), Value::Int(IntValue::U16(4))),
        ] {
            let err = block_on(gcd_builtin(left, right)).expect_err("invalid gcd inputs");
            assert_eq!(err.identifier(), GCD_ERROR_INVALID_INPUT.identifier);
        }
        let left = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let right = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let err = block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right)))
            .expect_err("shape mismatch");
        assert_eq!(err.identifier(), GCD_ERROR_SIZE_MISMATCH.identifier);
    }

    #[test]
    fn gcd_extended_outputs_preserve_signed_class_and_satisfy_bezout_identity() {
        let _guard = crate::output_count::push_output_count(Some(3));
        let left = Tensor::new_integer(IntegerStorage::I16(vec![30, -81]), vec![1, 2]).unwrap();
        let right = Tensor::new_integer(IntegerStorage::I16(vec![56, 57]), vec![1, 2]).unwrap();
        let Value::OutputList(outputs) =
            block_on(gcd_builtin(Value::Tensor(left), Value::Tensor(right))).unwrap()
        else {
            panic!("expected three outputs");
        };
        assert_eq!(outputs.len(), 3);
        let storages = outputs
            .iter()
            .map(|value| {
                let Value::Tensor(tensor) = value else {
                    panic!("expected tensor output")
                };
                tensor.integer_storage().cloned().expect("integer storage")
            })
            .collect::<Vec<_>>();
        let IntegerStorage::I16(gcds) = &storages[0] else {
            panic!("expected int16 gcd");
        };
        let IntegerStorage::I16(first) = &storages[1] else {
            panic!("expected int16 first coefficient");
        };
        let IntegerStorage::I16(second) = &storages[2] else {
            panic!("expected int16 second coefficient");
        };
        assert_eq!(gcds, &[2, 3]);
        for index in 0..2 {
            let left = [30i32, -81][index];
            let right = [56i32, 57][index];
            assert_eq!(
                left * i32::from(first[index]) + right * i32::from(second[index]),
                i32::from(gcds[index])
            );
        }
    }

    #[test]
    fn gcd_extended_outputs_reject_unsigned_classes() {
        let _guard = crate::output_count::push_output_count(Some(2));
        let err = block_on(gcd_builtin(
            Value::Int(IntValue::U16(30)),
            Value::Int(IntValue::U16(56)),
        ))
        .expect_err("unsigned extended gcd is unsupported");
        assert_eq!(err.identifier(), GCD_ERROR_INVALID_INPUT.identifier);
    }
}
