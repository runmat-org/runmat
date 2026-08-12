//! MATLAB-compatible `class` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::type_resolvers::class_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::class")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "class",
    op_kind: GpuOpKind::Custom("introspection"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Introspection-only builtin; providers do not need to implement hooks. RunMat reads residency metadata and returns a host string.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::class")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "class",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; class executes on the host and returns a string scalar.",
};

const CLASS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Class name for the input value.",
}];

const CLASS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to inspect.",
}];

const CLASS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "name = class(A)",
    inputs: &CLASS_INPUTS,
    outputs: &CLASS_OUTPUT,
}];

const CLASS_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const CLASS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CLASS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CLASS_ERRORS,
};

const CLASS_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer scalars and arrays report their exact signedness and width; gpuArray reports its container class without inspecting or gathering payload data.",
    }];

pub const CLASS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "name = class(A)",
        inputs: &CLASS_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The result is a host character class name. Dense, sparse, and paired complex integer host storage is read from exact dtype metadata; resident values return gpuArray without provider access.",
    }];

#[runtime_builtin(
    name = "class",
    category = "introspection",
    summary = "Return class names for values.",
    keywords = "class,type inspection,type name,gpuArray class",
    type_resolver(class_type),
    descriptor(crate::builtins::introspection::class::CLASS_DESCRIPTOR),
    integer_capabilities(crate::builtins::introspection::class::CLASS_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::introspection::class"
)]
fn class_builtin(value: Value) -> crate::BuiltinResult<String> {
    Ok(class_name_for_value(&value))
}

/// Return the canonical MATLAB class name for a runtime value.
pub(crate) fn class_name_for_value(value: &Value) -> String {
    match value {
        Value::Num(_) | Value::Complex(_, _) => "double".to_string(),
        Value::ComplexTensor(tensor) => tensor.numeric_dtype().class_name().to_string(),
        Value::Tensor(tensor) => tensor.numeric_dtype().class_name().to_string(),
        Value::SparseTensor(sparse) => sparse.class_name().to_string(),
        Value::Int(iv) => iv.class_name().to_string(),
        Value::Bool(_) | Value::LogicalArray(_) => "logical".to_string(),
        Value::String(_) | Value::StringArray(_) => "string".to_string(),
        Value::CharArray(_) => "char".to_string(),
        Value::Symbolic(_) | Value::SymbolicArray(_) => "sym".to_string(),
        Value::Cell(_) => "cell".to_string(),
        Value::Struct(_) => "struct".to_string(),
        Value::GpuTensor(_) => "gpuArray".to_string(),
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => "function_handle".to_string(),
        Value::HandleObject(handle) => {
            if handle.class_name.is_empty() {
                "handle".to_string()
            } else {
                handle.class_name.clone()
            }
        }
        Value::Listener(_) => "event.listener".to_string(),
        Value::ObjectArray(array) => array.class_name().to_string(),
        Value::Object(obj) => obj.class_name.clone(),
        Value::ClassRef(_) => "meta.class".to_string(),
        Value::MException(_) => "MException".to_string(),
        Value::OutputList(_) => "output_list".to_string(),
        Value::Future(_) => "parallel.Future".to_string(),
        Value::Task(_) => "parallel.Task".to_string(),
        Value::Pool(_) => "parallel.Pool".to_string(),
        Value::Job(_) => "parallel.Job".to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, IntValue, IntegerComplexStorage,
        IntegerStorage, Listener, LogicalArray, MException, ObjectInstance, StringArray,
        StructValue, SymbolicExpr, Tensor,
    };

    fn test_handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_double_for_numeric_scalars() {
        let name = class_builtin(Value::Num(std::f64::consts::PI)).expect("class");
        assert_eq!(name, "double");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_integer_type_names() {
        let name = class_builtin(Value::Int(IntValue::I32(12))).expect("class");
        assert_eq!(name, "int32");
    }

    #[test]
    fn class_reports_exact_integer_tensor_storage_type() {
        let tensor = Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX]),
            vec![1, 1],
        )
        .expect("uint64 tensor");

        assert_eq!(class_name_for_value(&Value::Tensor(tensor)), "uint64");
    }

    #[test]
    fn class_reports_exact_integer_sparse_storage_type() {
        let sparse = runmat_builtins::SparseTensor::new_integer(
            2,
            1,
            vec![0, 1],
            vec![1],
            IntegerStorage::U64(vec![u64::MAX]),
        )
        .expect("uint64 sparse");

        assert_eq!(class_name_for_value(&Value::SparseTensor(sparse)), "uint64");
    }

    #[test]
    fn class_reports_exact_integer_type_for_every_typed_complex_class() {
        let cases = [
            (
                "int8",
                IntegerStorage::I8(vec![-1]),
                IntegerStorage::I8(vec![2]),
            ),
            (
                "int16",
                IntegerStorage::I16(vec![-3]),
                IntegerStorage::I16(vec![4]),
            ),
            (
                "int32",
                IntegerStorage::I32(vec![-5]),
                IntegerStorage::I32(vec![6]),
            ),
            (
                "int64",
                IntegerStorage::I64(vec![-7]),
                IntegerStorage::I64(vec![8]),
            ),
            (
                "uint8",
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U8(vec![2]),
            ),
            (
                "uint16",
                IntegerStorage::U16(vec![3]),
                IntegerStorage::U16(vec![4]),
            ),
            (
                "uint32",
                IntegerStorage::U32(vec![5]),
                IntegerStorage::U32(vec![6]),
            ),
            (
                "uint64",
                IntegerStorage::U64(vec![7]),
                IntegerStorage::U64(vec![8]),
            ),
        ];

        for (expected, real, imag) in cases {
            let storage = IntegerComplexStorage::new(real, imag).expect("matching components");
            let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");
            assert_eq!(
                class_name_for_value(&Value::ComplexTensor(tensor)),
                expected
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_sym_for_symbolic_values() {
        let name = class_builtin(Value::Symbolic(SymbolicExpr::variable("x"))).expect("class");

        assert_eq!(name, "sym");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_expected_names_for_core_types() {
        let logical_scalar = Value::Bool(true);
        assert_eq!(class_name_for_value(&logical_scalar), "logical");

        let logical_array = Value::LogicalArray(
            LogicalArray::new(vec![1u8, 0u8, 1u8, 1u8], vec![2, 2]).expect("logical array"),
        );
        assert_eq!(class_name_for_value(&logical_array), "logical");

        let string_scalar = Value::String("hello".to_string());
        assert_eq!(class_name_for_value(&string_scalar), "string");

        let string_array = Value::StringArray(
            StringArray::new(vec!["Ada".into(), "Grace".into()], vec![1, 2]).expect("string array"),
        );
        assert_eq!(class_name_for_value(&string_array), "string");

        let char_array = Value::CharArray(CharArray::new_row("abc"));
        assert_eq!(class_name_for_value(&char_array), "char");

        let real_tensor = Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
        assert_eq!(class_name_for_value(&real_tensor), "double");

        let complex_scalar = Value::Complex(1.0, -1.0);
        assert_eq!(class_name_for_value(&complex_scalar), "double");

        let complex_tensor = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, -3.0)], vec![2, 1]).expect("complex tensor"),
        );
        assert_eq!(class_name_for_value(&complex_tensor), "double");

        let cell =
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Bool(false)], 1, 2).unwrap());
        assert_eq!(class_name_for_value(&cell), "cell");

        let mut st = StructValue::new();
        st.fields.insert("field".into(), Value::Num(42.0));
        let struct_value = Value::Struct(st);
        assert_eq!(class_name_for_value(&struct_value), "struct");

        let func_handle = Value::FunctionHandle("sin".into());
        assert_eq!(class_name_for_value(&func_handle), "function_handle");

        let closure = Value::Closure(Closure {
            function_name: "anon".into(),
            bound_function: None,
            captures: vec![],
        });
        assert_eq!(class_name_for_value(&closure), "function_handle");

        let class_ref = Value::ClassRef("pkg.Point".into());
        assert_eq!(class_name_for_value(&class_ref), "meta.class");

        let exception = Value::MException(MException::new("id:err".into(), "fail".into()));
        assert_eq!(class_name_for_value(&exception), "MException");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_gpuarray_without_gather() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let name = class_builtin(Value::GpuTensor(handle)).expect("class");
            assert_eq!(name, "gpuArray");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_handle_class_names() {
        let fallback = HandleRef {
            class_name: String::new(),
            target: test_handle_target(),
            valid: false,
        };
        let fallback_name = class_builtin(Value::HandleObject(fallback)).expect("class");
        assert_eq!(fallback_name, "handle");

        let handle = HandleRef {
            class_name: "MockHandle".into(),
            target: test_handle_target(),
            valid: true,
        };
        let name = class_builtin(Value::HandleObject(handle)).expect("class");
        assert_eq!(name, "MockHandle");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn class_reports_object_and_listener_classes() {
        let object = ObjectInstance::new("pkg.Point".into());
        let obj_name = class_builtin(Value::Object(object)).expect("class object");
        assert_eq!(obj_name, "pkg.Point");

        let listener = Listener {
            id: 1,
            target: test_handle_target(),
            target_class_name: "EventTarget".into(),
            event_name: "changed".into(),
            callback: test_handle_target(),
            enabled: true,
            valid: true,
        };
        let listener_name = class_builtin(Value::Listener(listener)).expect("class listener");
        assert_eq!(listener_name, "event.listener");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn class_reports_gpuarray_with_wgpu_provider() {
        use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
        use runmat_accelerate_api::AccelProvider;

        // Attempt to register a WGPU provider; skip if the environment lacks a compatible adapter.
        let provider = match ensure_wgpu_provider() {
            Ok(Some(p)) => p,
            _ => return,
        };

        let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("wgpu upload");
        let name = class_builtin(Value::GpuTensor(handle)).expect("class");
        assert_eq!(name, "gpuArray");
    }
}
