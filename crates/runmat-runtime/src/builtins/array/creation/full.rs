//! MATLAB-compatible `full` conversion for sparse matrix values.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, SparseTensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "full";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::full")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("storage-conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Converts host sparse matrices to dense host tensors; already-full GPU tensors remain resident.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::full")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Storage conversion executes outside numeric fusion.",
};

const FULL_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Full matrix or already-full input.",
}];

const FULL_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sparse or full matrix.",
}];

const FULL_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "A = full(S)",
    inputs: &FULL_INPUT,
    outputs: &FULL_OUTPUT,
}];

const FULL_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FULL.INVALID_INPUT",
    identifier: Some("RunMat:full:InvalidInput"),
    when: "Input is not a sparse matrix or already-full matrix value.",
    message: "full: invalid input",
};

const FULL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FULL.INTERNAL",
    identifier: Some("RunMat:full:Internal"),
    when: "Sparse-to-full materialisation fails internally.",
    message: "full: internal error",
};

const FULL_ERRORS: [BuiltinErrorDescriptor; 2] = [FULL_ERROR_INVALID_INPUT, FULL_ERROR_INTERNAL];

pub const FULL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FULL_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FULL_ERRORS,
};

fn full_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    full_error_with_message(format!("{}: {detail}", error.message), error)
}

fn full_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_input(type_name: &str) -> RuntimeError {
    full_error_with_detail(
        &FULL_ERROR_INVALID_INPUT,
        format!("conversion to full storage is not defined for {type_name}"),
    )
}

#[runtime_builtin(
    name = "full",
    category = "array/creation",
    summary = "Convert sparse matrix storage to full storage.",
    keywords = "full,sparse,dense,matrix,storage",
    accel = "storage-conversion",
    type_resolver(full_type),
    descriptor(crate::builtins::array::creation::full::FULL_DESCRIPTOR),
    builtin_path = "crate::builtins::array::creation::full"
)]
async fn full_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::SparseTensor(sparse) => full_from_sparse(sparse),
        passthrough @ (Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Complex(_, _)
        | Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::LogicalArray(_)
        | Value::CharArray(_)
        | Value::GpuTensor(_)) => Ok(passthrough),
        Value::String(_) | Value::StringArray(_) => Err(invalid_input("string")),
        Value::Symbolic(_) => Err(invalid_input("sym")),
        Value::Cell(_) => Err(invalid_input("cell")),
        Value::Struct(_) => Err(invalid_input("struct")),
        Value::Object(obj) => Err(invalid_input(&obj.class_name)),
        Value::HandleObject(handle) => Err(invalid_input(&handle.class_name)),
        Value::Listener(_) => Err(invalid_input("event.listener")),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => Err(invalid_input("function_handle")),
        Value::ClassRef(_) => Err(invalid_input("meta.class")),
        Value::MException(_) => Err(invalid_input("MException")),
        Value::OutputList(_) => Err(invalid_input("OutputList")),
    }
}

fn full_from_sparse(sparse: SparseTensor) -> BuiltinResult<Value> {
    if sparse.is_logical() {
        return sparse
            .to_dense_logical()
            .map(Value::LogicalArray)
            .map_err(|err| {
                full_error_with_detail(
                    &FULL_ERROR_INTERNAL,
                    format!("failed to densify sparse input: {err}"),
                )
            });
    }
    let tensor = sparse.to_dense().map_err(|err| {
        full_error_with_detail(
            &FULL_ERROR_INTERNAL,
            format!("failed to densify sparse input: {err}"),
        )
    })?;
    Ok(Value::Tensor(tensor))
}

fn full_type(args: &[Type], _context: &ResolveContext) -> Type {
    // RunMat does not currently have a sparse-specific `Type`; `Value::SparseTensor`
    // is represented as `Type::Tensor`, so preserving the input type still models
    // the dense tensor result of `full(S)`.
    args.first().cloned().unwrap_or(Type::Unknown)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CellArray, SparseTensor, Tensor, Value};

    fn run_full(value: Value) -> BuiltinResult<Value> {
        block_on(super::full_builtin(value))
    }

    fn expect_tensor(value: Value) -> Tensor {
        match value {
            Value::Tensor(tensor) => tensor,
            other => panic!("expected dense tensor, got {other:?}"),
        }
    }

    #[test]
    fn full_type_preserves_input_type() {
        let ty = Type::Tensor {
            shape: Some(vec![Some(2), Some(3)]),
        };
        assert_eq!(
            super::full_type(std::slice::from_ref(&ty), &ResolveContext::new(Vec::new())),
            ty
        );
        assert_eq!(
            super::full_type(&[], &ResolveContext::new(Vec::new())),
            Type::Unknown
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sparse_triplet_matrix_densifies_column_major() {
        let sparse =
            SparseTensor::new(3, 2, vec![0, 2, 3], vec![0, 2, 1], vec![10.0, 30.0, 20.0]).unwrap();
        let dense = expect_tensor(run_full(Value::SparseTensor(sparse)).unwrap());
        assert_eq!(dense.shape, vec![3, 2]);
        assert_eq!(
            dense.materialize_f64(),
            vec![10.0, 0.0, 30.0, 0.0, 20.0, 0.0]
        );
    }

    #[test]
    fn full_preserves_exact_uint64_sparse_storage() {
        let sparse = SparseTensor::new_integer(
            2,
            2,
            vec![0, 1, 2],
            vec![1, 0],
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX, 7]),
        )
        .expect("uint64 sparse");

        let dense = expect_tensor(run_full(Value::SparseTensor(sparse)).expect("full sparse"));
        assert_eq!(
            dense.integer_storage(),
            Some(&runmat_builtins::IntegerStorage::U64(vec![
                0,
                u64::MAX,
                7,
                0
            ]))
        );
    }

    #[test]
    fn full_preserves_logical_sparse_class_and_shape() {
        let sparse =
            SparseTensor::new_logical(3, 2, vec![0, 2, 3], vec![0, 2, 1]).expect("logical");
        let dense = run_full(Value::SparseTensor(sparse)).expect("full logical sparse");
        let Value::LogicalArray(dense) = dense else {
            panic!("expected logical array");
        };
        assert_eq!(dense.shape, vec![3, 2]);
        assert_eq!(dense.data, vec![1, 0, 1, 0, 1, 0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_sparse_matrix_densifies_to_zero_tensor() {
        let dense = expect_tensor(
            run_full(Value::SparseTensor(SparseTensor::zeros(2, 3))).expect("full sparse"),
        );
        assert_eq!(dense.shape, vec![2, 3]);
        assert_eq!(dense.materialize_f64(), vec![0.0; 6]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn already_full_numeric_values_pass_through() {
        let scalar = run_full(Value::Num(42.0)).unwrap();
        assert_eq!(scalar, Value::Num(42.0));

        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let result = run_full(Value::Tensor(tensor.clone())).unwrap();
        assert_eq!(result, Value::Tensor(tensor));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn already_full_gpu_tensor_remains_resident() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let handle = provider
                .upload(&HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload");
            let result = run_full(Value::GpuTensor(handle.clone())).expect("full gpu");
            assert_eq!(result, Value::GpuTensor(handle.clone()));
            provider.free(&handle).ok();
        });
    }

    #[test]
    fn sparse_dense_size_overflow_returns_internal_error() {
        let sparse = SparseTensor::zeros(usize::MAX, 2);
        let err = run_full(Value::SparseTensor(sparse)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:full:Internal"));
        assert!(err.message().contains("overflow"));
    }

    #[test]
    fn unsupported_values_raise_targeted_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = run_full(Value::Cell(cell)).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:full:InvalidInput"));
        assert!(err.message().contains("cell"));
    }
}
