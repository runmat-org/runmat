//! MATLAB-compatible `gt` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, LogicalArray, StringArray, Tensor, Value};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ProviderHook, ReductionNaN,
    ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::logical::rel::integer_comparison::{
    restore_explicit_comparison_result, select_comparison_output_source,
    try_complex_ordering_comparison, try_gpu_ordering_comparison, try_integer_comparison,
    IntegerComparisonError, IntegerComparisonOp,
};
use crate::builtins::logical::type_resolvers::symbolic_logical_binary_type;
use crate::builtins::math::symbolic::symbolic_named_binary;
use crate::{build_runtime_error, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::rel::gt")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gt",
    op_kind: GpuOpKind::Elementwise,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Binary {
        name: "elem_gt",
        commutative: false,
    }],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Prefers provider elem_gt kernels; complex-interleaved inputs compare provider-extracted real lanes, and unsupported routes gather to authoritative host storage.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::rel::gt")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gt",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |ctx: &FusionExprContext| {
            let lhs = ctx.inputs.first().ok_or(FusionError::MissingInput(0))?;
            let rhs = ctx.inputs.get(1).ok_or(FusionError::MissingInput(1))?;
            let (zero, one) = match ctx.scalar_ty {
                ScalarType::F32 => ("0.0", "1.0"),
                ScalarType::F64 => ("f64(0.0)", "f64(1.0)"),
                _ => return Err(FusionError::UnsupportedPrecision(ctx.scalar_ty)),
            };
            Ok(format!("select({zero}, {one}, ({lhs} > {rhs}))"))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Fusion emits comparison kernels that write 1 when the left operand is greater than the right.",
};

const BUILTIN_NAME: &str = "gt";

const GT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes compare exactly against integer, logical, single, and double operands.",
    },
    BuiltinIntegerInputCapability {
        name: "B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Mixed signed/unsigned and integer/double comparisons do not convert authoritative integer storage to f64.",
    },
];
pub const GT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = gt(integer_A,integer_or_numeric_B)",
        inputs: &GT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::BroadcastCompatible,
        notes: "Compatible implicit expansion is supported; correctness-first GPU fallback restores a logical result to the owning provider.",
    }];

const GT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical greater-than result.",
}];

const GT_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left operand.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right operand.",
    },
];

const GT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = gt(A, B)",
    inputs: &GT_INPUTS,
    outputs: &GT_OUTPUT,
}];

const GT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GT.INVALID_INPUT",
    identifier: Some("RunMat:gt:InvalidInput"),
    when: "Operands contain unsupported types or mixed numeric/string domains.",
    message: "gt: mixing numeric and string inputs is not supported",
};

const GT_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GT.SIZE_MISMATCH",
    identifier: Some("RunMat:gt:SizeMismatch"),
    when: "Operands are not broadcast-compatible.",
    message: "gt: array sizes are not compatible for broadcasting",
};

const GT_ERRORS: [BuiltinErrorDescriptor; 2] = [GT_ERROR_INVALID_INPUT, GT_ERROR_SIZE_MISMATCH];

pub const GT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &GT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &GT_ERRORS,
};

fn gt_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "gt",
    category = "logical/rel",
    summary = "Compute element-wise greater-than comparisons.",
    keywords = "gt,greater than,comparison,logical,gpu",
    accel = "elementwise",
    type_resolver(symbolic_logical_binary_type),
    descriptor(crate::builtins::logical::rel::gt::GT_DESCRIPTOR),
    integer_capabilities(crate::builtins::logical::rel::gt::GT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::logical::rel::gt"
)]
async fn gt_builtin(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let source = select_comparison_output_source(&lhs, &rhs, BUILTIN_NAME)?;
    match (&lhs, &rhs) {
        (Value::GpuTensor(a), Value::GpuTensor(b)) => {
            if runmat_accelerate_api::handle_integer_type(a).is_none()
                && runmat_accelerate_api::handle_integer_type(b).is_none()
            {
                if let Some(result) = try_gt_gpu(a, b).await {
                    return result;
                }
            }
        }
        (Value::GpuTensor(_), _) | (_, Value::GpuTensor(_)) => {}
        _ => {}
    }
    let result = gt_host(lhs, rhs).await?;
    restore_explicit_comparison_result(result, source.as_ref(), BUILTIN_NAME)
}

async fn try_gt_gpu(
    a: &GpuTensorHandle,
    b: &GpuTensorHandle,
) -> Option<crate::BuiltinResult<Value>> {
    try_gpu_ordering_comparison(a, b, IntegerComparisonOp::Gt).await
}

async fn gt_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    if let Some(result) = crate::builtins::table::categorical_compare(
        &lhs,
        &rhs,
        crate::builtins::table::CategoricalComparison::Gt,
    ) {
        return result;
    }

    if let Some(value) = symbolic_named_binary(&lhs, &rhs, BUILTIN_NAME) {
        return Ok(value);
    }

    let lhs = gather_gpu_operand(lhs)
        .await
        .map_err(|_| gt_error(&GT_ERROR_INVALID_INPUT))?;
    let rhs = gather_gpu_operand(rhs)
        .await
        .map_err(|_| gt_error(&GT_ERROR_INVALID_INPUT))?;
    let (lhs, rhs) = normalize_char_string(lhs, rhs);

    if let Some(result) = try_integer_comparison(&lhs, &rhs, IntegerComparisonOp::Gt).map_err(
        |error| match error {
            IntegerComparisonError::SizeMismatch => gt_error(&GT_ERROR_SIZE_MISMATCH),
            IntegerComparisonError::Internal => gt_error(&GT_ERROR_INVALID_INPUT),
        },
    )? {
        return Ok(result);
    }

    if let Some(result) = try_complex_ordering_comparison(&lhs, &rhs, IntegerComparisonOp::Gt)
        .map_err(|error| match error {
            IntegerComparisonError::SizeMismatch => gt_error(&GT_ERROR_SIZE_MISMATCH),
            IntegerComparisonError::Internal => gt_error(&GT_ERROR_INVALID_INPUT),
        })?
    {
        return Ok(result);
    }

    if let Some(result) = scalar_gt_value(&lhs, &rhs) {
        return result;
    }

    let left = GtOperand::from_value(lhs).await?;
    let right = GtOperand::from_value(rhs).await?;

    match (left, right) {
        (GtOperand::Numeric(a), GtOperand::Numeric(b)) => {
            let (data, shape) = numeric_gt(&a, &b)?;
            logical_result(data, shape)
        }
        (GtOperand::String(a), GtOperand::String(b)) => {
            let (data, shape) = string_gt(&a, &b)?;
            logical_result(data, shape)
        }
        (GtOperand::Numeric(_), GtOperand::String(_))
        | (GtOperand::String(_), GtOperand::Numeric(_)) => Err(gt_error(&GT_ERROR_INVALID_INPUT)),
    }
}

async fn gather_gpu_operand(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value).await,
        _ => Ok(value),
    }
}

fn scalar_numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Some(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.rows * ca.cols == 1 => {
            Some(ca.data.first().map(|&ch| ch as u32 as f64).unwrap_or(0.0))
        }
        _ => None,
    }
}

fn scalar_string_value(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data.first().cloned(),
        _ => None,
    }
}

fn scalar_gt_value(lhs: &Value, rhs: &Value) -> Option<crate::BuiltinResult<Value>> {
    let left_string = scalar_string_value(lhs);
    let right_string = scalar_string_value(rhs);
    if left_string.is_some() || right_string.is_some() {
        let left = left_string?;
        let right = right_string?;
        return Some(Ok(Value::Bool(left > right)));
    }

    let left = scalar_numeric_value(lhs)?;
    let right = scalar_numeric_value(rhs)?;
    Some(Ok(Value::Bool(left > right)))
}

fn normalize_char_string(lhs: Value, rhs: Value) -> (Value, Value) {
    match (lhs, rhs) {
        (Value::CharArray(ca), Value::String(s)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(text), Value::String(s))
        }
        (Value::String(s), Value::CharArray(ca)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(s), Value::String(text))
        }
        (Value::CharArray(ca), Value::StringArray(sa)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::String(text), Value::StringArray(sa))
        }
        (Value::StringArray(sa), Value::CharArray(ca)) => {
            let text: String = ca.data.into_iter().collect();
            (Value::StringArray(sa), Value::String(text))
        }
        (lhs, rhs) => (lhs, rhs),
    }
}

fn logical_result(data: Vec<u8>, shape: Vec<usize>) -> crate::BuiltinResult<Value> {
    if tensor::element_count(&shape) <= 1 && data.len() == 1 {
        Ok(Value::Bool(data[0] != 0))
    } else {
        LogicalArray::new(data, shape)
            .map(Value::LogicalArray)
            .map_err(|_| gt_error(&GT_ERROR_INVALID_INPUT))
    }
}

enum GtOperand {
    Numeric(NumericBuffer),
    String(StringBuffer),
}

impl GtOperand {
    async fn from_value(value: Value) -> crate::BuiltinResult<Self> {
        match value {
            Value::Num(n) => Ok(GtOperand::Numeric(NumericBuffer::scalar(n))),
            Value::Bool(flag) => Ok(GtOperand::Numeric(NumericBuffer::scalar(if flag {
                1.0
            } else {
                0.0
            }))),
            Value::Int(i) => Ok(GtOperand::Numeric(NumericBuffer::scalar(i.to_f64()))),
            Value::Tensor(tensor) => Ok(GtOperand::Numeric(NumericBuffer::from_tensor(tensor))),
            Value::LogicalArray(array) => {
                Ok(GtOperand::Numeric(NumericBuffer::from_logical(array)))
            }
            Value::CharArray(array) => {
                Ok(GtOperand::Numeric(NumericBuffer::from_char_array(array)))
            }
            Value::String(s) => Ok(GtOperand::String(StringBuffer::scalar(s))),
            Value::StringArray(sa) => Ok(GtOperand::String(StringBuffer::from_array(sa))),
            Value::GpuTensor(handle) => {
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|_| gt_error(&GT_ERROR_INVALID_INPUT))?;
                Ok(GtOperand::Numeric(NumericBuffer::from_tensor(tensor)))
            }
            Value::Complex(_, _) | Value::ComplexTensor(_) => {
                Err(gt_error(&GT_ERROR_INVALID_INPUT))
            }
            _ => Err(gt_error(&GT_ERROR_INVALID_INPUT)),
        }
    }
}

fn numeric_gt(
    lhs: &NumericBuffer,
    rhs: &NumericBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|_| gt_error(&GT_ERROR_SIZE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            0.0
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset]
        };
        let rhs_val = if rhs.data.is_empty() {
            0.0
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset]
        };
        out.push(if lhs_val > rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

fn string_gt(
    lhs: &StringBuffer,
    rhs: &StringBuffer,
) -> crate::BuiltinResult<(Vec<u8>, Vec<usize>)> {
    let shape = broadcast_shapes(BUILTIN_NAME, &lhs.shape, &rhs.shape)
        .map_err(|_| gt_error(&GT_ERROR_SIZE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return Ok((Vec::new(), shape));
    }
    let strides_l = compute_strides(&lhs.shape);
    let strides_r = compute_strides(&rhs.shape);
    let mut out = Vec::with_capacity(total);
    for idx in 0..total {
        let lhs_val = if lhs.data.is_empty() {
            ""
        } else {
            let offset = broadcast_index(idx, &shape, &lhs.shape, &strides_l);
            lhs.data[offset].as_str()
        };
        let rhs_val = if rhs.data.is_empty() {
            ""
        } else {
            let offset = broadcast_index(idx, &shape, &rhs.shape, &strides_r);
            rhs.data[offset].as_str()
        };
        out.push(if lhs_val > rhs_val { 1 } else { 0 });
    }
    Ok((out, shape))
}

#[derive(Debug)]
struct NumericBuffer {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl NumericBuffer {
    fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            shape: vec![1, 1],
        }
    }

    fn from_tensor(tensor: Tensor) -> Self {
        let shape = tensor.shape.clone();
        Self {
            data: tensor::tensor_into_values_f64(tensor),
            shape,
        }
    }

    fn from_logical(array: LogicalArray) -> Self {
        let shape = array.shape.clone();
        let data = array
            .data
            .into_iter()
            .map(|b| if b != 0 { 1.0 } else { 0.0 })
            .collect();
        Self { data, shape }
    }

    fn from_char_array(array: CharArray) -> Self {
        let rows = array.rows;
        let cols = array.cols;
        if rows == 0 || cols == 0 {
            return Self {
                data: Vec::new(),
                shape: vec![rows, cols],
            };
        }
        let mut data = Vec::with_capacity(rows * cols);
        for c in 0..cols {
            for r in 0..rows {
                let idx = r * cols + c;
                let ch = array.data[idx];
                data.push(ch as u32 as f64);
            }
        }
        Self {
            data,
            shape: vec![rows, cols],
        }
    }
}

#[derive(Debug)]
struct StringBuffer {
    data: Vec<String>,
    shape: Vec<usize>,
}

impl StringBuffer {
    fn scalar(value: String) -> Self {
        Self {
            data: vec![value],
            shape: vec![1, 1],
        }
    }

    fn from_array(array: StringArray) -> Self {
        Self {
            data: array.data,
            shape: array.shape,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;

    fn run_gt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        block_on(super::gt_builtin(lhs, rhs))
    }

    #[test]
    fn scalar_numeric_value_reads_typed_integer_tensor_storage_exactly() {
        let tensor = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993]),
            vec![1, 1],
        )
        .expect("integer tensor");

        assert_eq!(
            scalar_numeric_value(&Value::Tensor(tensor)),
            Some(9_007_199_254_740_993_u64 as f64)
        );
    }

    #[test]
    fn gt_dense_integer_arrays_read_exact_storage_without_mirror() {
        let lhs = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]),
            vec![2, 1],
        )
        .expect("lhs");
        let rhs = Tensor::new_integer(
            runmat_value::IntegerStorage::I64(vec![0, 1, i64::MAX]),
            vec![1, 3],
        )
        .expect("rhs");

        let result = run_gt(Value::Tensor(lhs), Value::Tensor(rhs)).expect("gt");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![2, 3]);
                assert_eq!(array.data, vec![0, 1, 0, 1, 0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn gt_integer_contract_is_exact_broadcasting_to_logical() {
        assert_eq!(GT_INTEGER_CAPABILITIES.len(), 1);
        assert_eq!(GT_INTEGER_CAPABILITIES[0].inputs.len(), 2);
        assert!(GT_INTEGER_CAPABILITIES[0]
            .inputs
            .iter()
            .all(|input| input.classes.len() == 8));
        assert_eq!(
            GT_INTEGER_CAPABILITIES[0].computation_domain,
            BuiltinIntegerComputationDomain::Predicate
        );
        assert_eq!(
            GT_INTEGER_CAPABILITIES[0].output_class,
            BuiltinIntegerOutputClassRule::Logical
        );
        assert_eq!(
            GT_INTEGER_CAPABILITIES[0].overload,
            BuiltinIntegerOverloadKind::BroadcastCompatible
        );
    }

    #[test]
    fn gt_gpu_wide_integer_scalar_fallback_is_exact_owner_resident_logical() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new_integer(
                runmat_value::IntegerStorage::U64(vec![
                    9_007_199_254_740_992,
                    9_007_199_254_740_993,
                ]),
                vec![1, 2],
            )
            .expect("wide integer lhs");
            let handle = gpu_helpers::upload_tensor(provider, &lhs).expect("upload exact lhs");
            let result = run_gt(
                Value::GpuTensor(handle),
                Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_992)),
            )
            .expect("resident exact gt");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident logical result");
            };
            assert!(runmat_accelerate_api::handle_is_logical(result_handle));
            assert!(std::ptr::eq(
                runmat_accelerate_api::provider_for_handle(result_handle).expect("result owner"),
                provider
            ));
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.shape, vec![1, 2]);
            assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0]);
        });
    }

    #[test]
    fn gt_resident_single_does_not_round_a_double_scalar_for_comparison() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::from_f32(vec![16_777_216.0], vec![1, 1]).expect("single lhs");
            let handle = gpu_helpers::upload_tensor(provider, &lhs).expect("upload single lhs");
            let result = run_gt(Value::GpuTensor(handle), Value::Num(16_777_217.0))
                .expect("mixed-precision gt");
            let Value::GpuTensor(result_handle) = &result else {
                panic!("expected resident logical result");
            };
            assert!(runmat_accelerate_api::handle_is_logical(result_handle));
            assert!(std::ptr::eq(
                runmat_accelerate_api::provider_for_handle(result_handle).expect("result owner"),
                provider
            ));
            let gathered = test_support::gather(result).expect("gather result");
            assert_eq!(gathered.materialize_f64(), vec![0.0]);
        });
    }

    #[cfg(feature = "wgpu")]
    fn run_gt_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
        block_on(gt_host(lhs, rhs))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_scalar_true() {
        let result = run_gt(Value::Num(5.0), Value::Num(4.0)).expect("gt");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_scalar_false() {
        let result = run_gt(Value::Num(2.0), Value::Num(3.0)).expect("gt");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_vector_broadcast() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![1, 4]).unwrap();
        let result = run_gt(Value::Tensor(tensor), Value::Num(2.0)).expect("gt");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 4]);
                assert_eq!(array.data, vec![0, 1, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_char_array_against_numeric() {
        let chars = CharArray::new(vec!['A', 'B', 'C'], 1, 3).unwrap();
        let tensor = Tensor::new(vec![65.0, 65.0, 65.0], vec![1, 3]).unwrap();
        let result = run_gt(Value::CharArray(chars), Value::Tensor(tensor)).expect("gt");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![1, 3]);
                assert_eq!(array.data, vec![0, 1, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_string_array_against_scalar() {
        let array = StringArray::new(vec!["apple".into(), "carrot".into()], vec![1, 2]).unwrap();
        let result = run_gt(Value::StringArray(array), Value::String("banana".into())).expect("gt");
        match result {
            Value::LogicalArray(mask) => {
                assert_eq!(mask.shape, vec![1, 2]);
                assert_eq!(mask.data, vec![0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_string_numeric_error() {
        let err =
            run_gt(Value::String("apple".into()), Value::Num(3.0)).expect_err("expected error");
        assert!(err.message().contains("mixing numeric and string"));
        assert_eq!(err.identifier(), GT_ERROR_INVALID_INPUT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_complex_compares_real_component() {
        let result = run_gt(Value::Complex(3.0, -99.0), Value::Num(2.0)).expect("gt");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gt_gpu_provider_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
            let rhs = Tensor::new(vec![0.0, 5.0, 6.0], vec![1, 3]).unwrap();
            let view_l = HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            };
            let view_r = HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            };
            let handle_l = provider.upload(&view_l).expect("upload lhs");
            let handle_r = provider.upload(&view_r).expect("upload rhs");
            let result =
                run_gt(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).expect("gt");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 1.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gt_wgpu_matches_host() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let lhs = Tensor::new(vec![0.0, 2.0, 5.0, 8.0], vec![4, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0, 1.5, 4.0, 8.0], vec![4, 1]).unwrap();
        let cpu = run_gt_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();

        let view_l = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };
        let provider = runmat_accelerate_api::provider().expect("provider");
        let handle_l = provider.upload(&view_l).expect("upload lhs");
        let handle_r = provider.upload(&view_r).expect("upload rhs");
        let gpu = run_gt(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).unwrap();
        let gathered = test_support::gather(gpu).expect("gather");

        match (cpu, gathered) {
            (Value::LogicalArray(host), tensor) => {
                assert_eq!(tensor.shape, host.shape);
                let expected: Vec<f64> = host
                    .data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect();
                assert_eq!(tensor.materialize_f64(), expected);
            }
            (Value::Bool(host_flag), tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                let expected = if host_flag { 1.0 } else { 0.0 };
                assert_eq!(tensor.materialize_f64(), vec![expected]);
            }
            other => panic!("unexpected output combination: {other:?}"),
        }
    }
}
