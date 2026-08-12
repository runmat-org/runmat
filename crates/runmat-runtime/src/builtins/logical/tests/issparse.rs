//! MATLAB-compatible `issparse` builtin for RunMat sparse matrix values.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::BuiltinResult;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::issparse")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "issparse",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Reports whether the value is a host sparse matrix; dense gpuArray handles are not sparse.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::issparse")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "issparse",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that executes outside fusion and returns a scalar logical.",
};

const ISSPARSE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input is a sparse matrix.",
}];

const ISSPARSE_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to test.",
}];

const ISSPARSE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = issparse(A)",
    inputs: &ISSPARSE_INPUTS,
    outputs: &ISSPARSE_OUTPUT,
}];

const ISSPARSE_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISSPARSE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISSPARSE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISSPARSE_ERRORS,
};

#[runtime_builtin(
    name = "issparse",
    category = "logical/tests",
    summary = "Return true when a value is a sparse matrix.",
    keywords = "issparse,sparse,matrix,type,logical",
    accel = "metadata",
    type_resolver(bool_scalar_type),
    descriptor(crate::builtins::logical::tests::issparse::ISSPARSE_DESCRIPTOR),
    builtin_path = "crate::builtins::logical::tests::issparse"
)]
async fn issparse_builtin(value: Value) -> BuiltinResult<Value> {
    Ok(Value::Bool(matches!(value, Value::SparseTensor(_))))
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
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, IntValue, LogicalArray, MException, ObjectInstance,
        SparseTensor, StringArray, StructValue, SymbolicExpr, Tensor, Value,
    };

    fn run_issparse(value: Value) -> BuiltinResult<Value> {
        block_on(super::issparse_builtin(value))
    }

    #[test]
    fn issparse_type_returns_bool() {
        assert_eq!(
            super::bool_scalar_type(&[Type::Unknown], &ResolveContext::new(Vec::new())),
            Type::Bool
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sparse_tensors_report_true() {
        let sparse = SparseTensor::new(3, 2, vec![0, 1, 2], vec![1, 2], vec![4.0, -1.0]).unwrap();
        assert_eq!(
            run_issparse(Value::SparseTensor(sparse)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            run_issparse(Value::SparseTensor(SparseTensor::zeros(4, 5))).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn dense_numeric_and_logical_values_report_false() {
        assert_eq!(
            run_issparse(Value::Int(IntValue::I32(1))).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(run_issparse(Value::Num(1.0)).unwrap(), Value::Bool(false));
        assert_eq!(
            run_issparse(Value::Complex(1.0, -2.0)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(run_issparse(Value::Bool(true)).unwrap(), Value::Bool(false));
        assert_eq!(
            run_issparse(Value::LogicalArray(
                LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Tensor(
                Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()
            ))
            .unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap()
            ))
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn containers_text_and_objects_report_false() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let string = Value::String("abc".into());
        let chars = CharArray::new_row("abc");
        let strings = StringArray::new(vec!["abc".into()], vec![1, 1]).unwrap();
        let structure = StructValue::new();
        let object = ObjectInstance::new("Example".into());
        assert_eq!(run_issparse(Value::Cell(cell)).unwrap(), Value::Bool(false));
        assert_eq!(run_issparse(string).unwrap(), Value::Bool(false));
        assert_eq!(
            run_issparse(Value::CharArray(chars)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::StringArray(strings)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Struct(structure)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Object(object)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::FunctionHandle("sin".into())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::Symbolic(SymbolicExpr::variable("x"))).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::ClassRef("Example".into())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            run_issparse(Value::MException(MException::new(
                "RunMat:test".into(),
                "not sparse".into()
            )))
            .unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_handles_report_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = run_issparse(Value::GpuTensor(handle.clone())).expect("issparse");
            assert_eq!(result, Value::Bool(false));
            provider.free(&handle).ok();
        });
    }
}
