//! MATLAB-compatible `isgpuarray` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::BuiltinResult;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isgpuarray")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isgpuarray",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Reports whether the value is a gpuArray without gathering device buffers.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isgpuarray")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isgpuarray",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that executes outside of fusion pipelines.",
};

const ISGPUARRAY_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input is a gpuArray handle.",
}];

const ISGPUARRAY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test.",
}];

const ISGPUARRAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isgpuarray(A)",
    inputs: &ISGPUARRAY_INPUTS,
    outputs: &ISGPUARRAY_OUTPUT,
}];

const ISGPUARRAY_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISGPUARRAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISGPUARRAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISGPUARRAY_ERRORS,
};

const ISGPUARRAY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An explicitly constructed gpuArray may contain any of the eight integer classes; the predicate inspects residency intent without downloading its payload.",
    }];

pub const ISGPUARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "tf = isgpuarray(integer_gpuArray)",
        inputs: &ISGPUARRAY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The metadata query returns true for explicit integer gpuArray values and false for host integers or RunMat-internal automatic residency; it performs no gather or numeric conversion.",
    }];

#[runtime_builtin(
    name = "isgpuarray",
    category = "logical/tests",
    summary = "Return true when a value is stored as a gpuArray handle.",
    keywords = "isgpuarray,gpuarray,gpu,type,logical",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    descriptor(crate::builtins::logical::tests::isgpuarray::ISGPUARRAY_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::logical::tests::isgpuarray::ISGPUARRAY_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::logical::tests::isgpuarray"
)]
async fn isgpuarray_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(&handle),
        _ => false,
    }))
}

fn bool_scalar_type(_: &[Type], _context: &ResolveContext) -> Type {
    Type::Bool
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{Tensor, Value};

    fn run_isgpuarray(value: Value) -> BuiltinResult<Value> {
        block_on(super::isgpuarray_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_gpu_values_report_false() {
        assert_eq!(run_isgpuarray(Value::Num(1.0)).unwrap(), Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn only_explicit_gpuarray_handles_report_true() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let automatic =
                run_isgpuarray(Value::GpuTensor(handle.clone())).expect("isgpuarray automatic");
            assert_eq!(automatic, Value::Bool(false));

            runmat_accelerate_api::mark_handle_explicit(&handle);
            let explicit =
                run_isgpuarray(Value::GpuTensor(handle.clone())).expect("isgpuarray explicit");
            assert_eq!(explicit, Value::Bool(true));
            provider.free(&handle).ok();
        });
    }
}
