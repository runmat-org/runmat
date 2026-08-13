//! MATLAB-compatible `isa` builtin with GPU-aware semantics for RunMat.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::introspection::class::class_name_for_value;
use crate::builtins::introspection::type_resolvers::isa_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{handle_integer_type, handle_is_logical};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::introspection::isa")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isa",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::InheritInputs,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Metadata predicate that returns host logical scalars; no GPU kernels or gathers are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::introspection::isa")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isa",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Not eligible for fusion planning; isa executes on the host and produces a logical scalar.",
};

const ISA_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input belongs to the requested class/category.",
}];

const ISA_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input value to inspect.",
    },
    BuiltinParamDescriptor {
        name: "type_name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Class or abstract type name.",
    },
];

const ISA_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = isa(A, type_name)",
    inputs: &ISA_INPUTS,
    outputs: &ISA_OUTPUT,
}];

const BUILTIN_NAME: &str = "isa";

const ISA_ERROR_TYPE_NAME_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ISA.TYPE_NAME_INVALID",
    identifier: None,
    when: "Second argument is not a string scalar or row character vector.",
    message: "isa: TYPE must be a string scalar or character vector",
};

const ISA_ERRORS: [BuiltinErrorDescriptor; 1] = [ISA_ERROR_TYPE_NAME_INVALID];

pub const ISA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISA_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISA_ERRORS,
};

#[runtime_builtin(
    name = "isa",
    category = "introspection",
    summary = "Test whether a value belongs to a specified class or category.",
    keywords = "isa,type checking,class comparison,numeric category,gpuArray",
    accel = "metadata",
    type_resolver(isa_type),
    descriptor(crate::builtins::introspection::isa::ISA_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::isa"
)]
fn isa_builtin(value: Value, class_designator: Value) -> crate::BuiltinResult<Value> {
    let type_name = parse_type_name(&class_designator)?;
    let result = value_is_a(&value, &type_name);
    Ok(Value::Bool(result))
}

fn parse_type_name(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) => {
            if sa.rows == 1 && sa.cols == 1 && !sa.data.is_empty() {
                Ok(sa.data[0].clone())
            } else {
                Err(isa_error(&ISA_ERROR_TYPE_NAME_INVALID).into())
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ca.data.iter().collect())
            } else {
                Err(isa_error(&ISA_ERROR_TYPE_NAME_INVALID).into())
            }
        }
        _ => Err(isa_error(&ISA_ERROR_TYPE_NAME_INVALID).into()),
    }
}

fn isa_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    let mut builder = build_runtime_error(error.message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn value_is_a(value: &Value, requested: &str) -> bool {
    let trimmed = requested.trim();
    if trimmed.is_empty() {
        return false;
    }
    let requested_lower = trimmed.to_ascii_lowercase();
    match requested_lower.as_str() {
        "numeric" => is_numeric(value),
        "float" => is_float(value),
        "integer" => is_integer(value),
        "logical" => is_logical(value),
        "char" => matches!(value, Value::CharArray(_)),
        "string" => matches!(value, Value::String(_) | Value::StringArray(_)),
        "cell" => matches!(value, Value::Cell(_)),
        "struct" => matches!(value, Value::Struct(_)),
        "function_handle" => matches!(
            value,
            Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle { .. }
                | Value::Closure(_)
        ),
        "gpuarray" => matches!(value, Value::GpuTensor(_)),
        "listener" | "event.listener" => matches!(value, Value::Listener(_)),
        "meta.class" => matches!(value, Value::ClassRef(_)),
        "mexception" => matches!(value, Value::MException(_)),
        "handle" => is_handle_like(value),
        _ => {
            let actual = class_name_for_value(value);
            if actual.eq_ignore_ascii_case(trimmed) {
                return true;
            }
            match value {
                Value::ObjectArray(array) => class_inherits(array.class_name(), &requested_lower),
                Value::Object(obj) => class_inherits(&obj.class_name, &requested_lower),
                Value::HandleObject(handle) => {
                    !handle.class_name.is_empty()
                        && class_inherits(&handle.class_name, &requested_lower)
                }
                _ => false,
            }
        }
    }
}

fn is_numeric(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Tensor(_)
        | Value::ComplexTensor(_)
        | Value::Complex(_, _)
        | Value::Int(_) => true,
        Value::SparseTensor(sparse) => !sparse.is_logical(),
        Value::GpuTensor(handle) => !handle_is_logical(handle),
        _ => false,
    }
}

fn is_float(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_none(),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none(),
        Value::SparseTensor(sparse) => !sparse.is_logical() && sparse.integer_storage().is_none(),
        Value::GpuTensor(handle) => {
            !handle_is_logical(handle) && handle_integer_type(handle).is_none()
        }
        _ => false,
    }
}

fn is_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_some(),
        Value::SparseTensor(sparse) => sparse.integer_storage().is_some(),
        Value::GpuTensor(handle) => handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn is_logical(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::SparseTensor(sparse) => sparse.is_logical(),
        Value::GpuTensor(handle) => handle_is_logical(handle),
        _ => false,
    }
}

fn is_handle_like(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::ObjectArray(array) => class_inherits(array.class_name(), "handle"),
        Value::Object(obj) => class_inherits(&obj.class_name, "handle"),
        _ => false,
    }
}

fn class_inherits(class_name: &str, requested_lower: &str) -> bool {
    if class_name.eq_ignore_ascii_case(requested_lower) {
        return true;
    }
    let mut cursor = Some(class_name.to_string());
    let mut visited = std::collections::HashSet::new();
    while let Some(name) = cursor {
        if !visited.insert(name.clone()) {
            break;
        }
        if name.eq_ignore_ascii_case(requested_lower) {
            return true;
        }
        if let Some(def) = crate::class_registry::get_class(&name) {
            cursor = def.parent.clone();
        } else {
            break;
        }
    }
    false
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::{gpu_helpers, test_support};
    use crate::class_registry::RuntimeClass;
    use runmat_accelerate_api::{
        AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostTensorView,
    };
    use runmat_value::{
        CellArray, CharArray, ComplexTensor, HandleRef, IntValue, IntegerComplexStorage,
        IntegerStorage, Listener, LogicalArray, ObjectInstance, SparseTensor, StringArray,
        StructValue, Tensor,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_CLASS_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_handle_target() -> runmat_gc::GcHandle {
        runmat_gc::gc_allocate(Value::Num(0.0)).expect("gc allocation")
    }

    fn unique_class_name(prefix: &str) -> String {
        let id = TEST_CLASS_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("{}_{}", prefix, id)
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_reports_expected_results_for_doubles() {
        let double_result = isa_builtin(Value::Num(42.0), Value::from("double")).expect("isa");
        assert_eq!(double_result, Value::Bool(true));

        let numeric_result = isa_builtin(Value::Num(42.0), Value::from("numeric")).expect("isa");
        assert_eq!(numeric_result, Value::Bool(true));

        let integer_result = isa_builtin(Value::Num(42.0), Value::from("integer")).expect("isa");
        assert_eq!(integer_result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_integer_category_matches_int_values() {
        let value = Value::Int(IntValue::I16(12));
        let int_result = isa_builtin(value.clone(), Value::from("integer")).expect("isa");
        assert_eq!(int_result, Value::Bool(true));

        let float_result = isa_builtin(value, Value::from("float")).expect("isa");
        assert_eq!(float_result, Value::Bool(false));
    }

    #[test]
    fn isa_integer_category_matches_exact_integer_arrays() {
        let dense = Value::Tensor(
            Tensor::new_integer(
                IntegerStorage::U64(vec![u64::MAX, 9_007_199_254_740_993]),
                vec![1, 2],
            )
            .expect("uint64 tensor"),
        );
        assert_eq!(
            isa_builtin(dense.clone(), Value::from("integer")).expect("isa"),
            Value::Bool(true)
        );
        assert_eq!(
            isa_builtin(dense.clone(), Value::from("float")).expect("isa"),
            Value::Bool(false)
        );
        assert_eq!(
            isa_builtin(dense, Value::from("uint64")).expect("isa"),
            Value::Bool(true)
        );

        let sparse = Value::SparseTensor(
            SparseTensor::new_integer(
                2,
                1,
                vec![0, 1],
                vec![1],
                IntegerStorage::I64(vec![i64::MIN]),
            )
            .expect("int64 sparse"),
        );
        assert_eq!(
            isa_builtin(sparse.clone(), Value::from("integer")).expect("isa"),
            Value::Bool(true)
        );
        assert_eq!(
            isa_builtin(sparse.clone(), Value::from("float")).expect("isa"),
            Value::Bool(false)
        );
        assert_eq!(
            isa_builtin(sparse, Value::from("int64")).expect("isa"),
            Value::Bool(true)
        );

        let typed_complex = Value::ComplexTensor(
            ComplexTensor::new_integer(
                IntegerComplexStorage::new(
                    IntegerStorage::I16(vec![-1, i16::MAX]),
                    IntegerStorage::I16(vec![2, i16::MIN]),
                )
                .expect("matching complex integer storage"),
                vec![1, 2],
            )
            .expect("typed complex integer tensor"),
        );
        assert_eq!(
            isa_builtin(typed_complex.clone(), Value::from("integer")).expect("isa"),
            Value::Bool(true)
        );
        assert_eq!(
            isa_builtin(typed_complex.clone(), Value::from("float")).expect("isa"),
            Value::Bool(false)
        );
        assert_eq!(
            isa_builtin(typed_complex, Value::from("int16")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[test]
    fn isa_float_category_excludes_only_integer_backed_numeric_arrays() {
        let dense_float =
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("double tensor"));
        assert_eq!(
            isa_builtin(dense_float, Value::from("float")).expect("isa"),
            Value::Bool(true)
        );

        let sparse_float = Value::SparseTensor(
            SparseTensor::new(2, 1, vec![0, 1], vec![1], vec![2.5]).expect("double sparse"),
        );
        assert_eq!(
            isa_builtin(sparse_float, Value::from("float")).expect("isa"),
            Value::Bool(true)
        );

        let complex_float = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).expect("double complex tensor"),
        );
        assert_eq!(
            isa_builtin(complex_float, Value::from("float")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_handles_logical_and_char_types() {
        let logical = Value::LogicalArray(LogicalArray::new(vec![1], vec![1]).unwrap());
        assert_eq!(
            isa_builtin(logical, Value::from("logical")).expect("isa"),
            Value::Bool(true)
        );

        let char_array = Value::CharArray(CharArray::new_row("RunMat"));
        assert_eq!(
            isa_builtin(char_array, Value::from("char")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_string_and_struct_detection() {
        let string_scalar = Value::String("runmat".into());
        assert_eq!(
            isa_builtin(string_scalar, Value::from("string")).expect("isa"),
            Value::Bool(true)
        );

        let mut st = StructValue::new();
        st.fields.insert("field".into(), Value::Num(1.0));
        assert_eq!(
            isa_builtin(Value::Struct(st), Value::from("struct")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_cell_and_function_handle() {
        let cell = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap());
        assert_eq!(
            isa_builtin(cell, Value::from("cell")).expect("isa"),
            Value::Bool(true)
        );

        let func = Value::FunctionHandle("sin".into());
        assert_eq!(
            isa_builtin(func, Value::from("function_handle")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_gpu_arrays_treat_metadata_correctly() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu_value = Value::GpuTensor(handle);

            let numeric = isa_builtin(gpu_value.clone(), Value::from("numeric")).expect("isa");
            assert_eq!(numeric, Value::Bool(true));

            let double = isa_builtin(gpu_value, Value::from("double")).expect("isa");
            assert_eq!(double, Value::Bool(false));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_gpu_logical_handles_match_categories() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let logical_value = gpu_helpers::logical_gpu_value(handle.clone());

            let logical = isa_builtin(logical_value.clone(), Value::from("logical")).expect("isa");
            assert_eq!(logical, Value::Bool(true));

            let numeric =
                isa_builtin(logical_value, Value::from("numeric")).expect("isa numeric false");
            assert_eq!(numeric, Value::Bool(false));
        });
    }

    #[test]
    fn isa_gpu_integer_handles_match_integer_category() {
        test_support::with_test_provider(|provider| {
            let values = [u64::MAX, 9_007_199_254_740_993];
            let shape = [1usize, 2usize];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&values),
                    shape: &shape,
                })
                .expect("upload integer gpu tensor");
            let gpu_value = Value::GpuTensor(handle);

            assert_eq!(
                isa_builtin(gpu_value.clone(), Value::from("numeric")).expect("isa numeric"),
                Value::Bool(true)
            );
            assert_eq!(
                isa_builtin(gpu_value.clone(), Value::from("integer")).expect("isa integer"),
                Value::Bool(true)
            );
            assert_eq!(
                isa_builtin(gpu_value, Value::from("float")).expect("isa float"),
                Value::Bool(false)
            );
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_handle_aliases_and_inheritance() {
        let handle = HandleRef {
            class_name: "TestHandle".into(),
            target: test_handle_target(),
            valid: true,
        };
        assert_eq!(
            isa_builtin(Value::HandleObject(handle), Value::from("handle")).expect("isa"),
            Value::Bool(true)
        );

        // Register a class that derives from handle and ensure inheritance is respected.
        let class_name = "pkg.TestHandle";
        let def = crate::class_registry::RuntimeClass {
            name: class_name.into(),
            parent: Some("handle".into()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        };
        crate::class_registry::register_class(def);
        let obj = Value::Object(ObjectInstance::new(class_name.into()));
        let handle_result = isa_builtin(obj.clone(), Value::from("handle")).expect("isa");
        assert_eq!(handle_result, Value::Bool(true));
        let exact = isa_builtin(obj, Value::from(class_name)).expect("isa");
        assert_eq!(exact, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_inheritance_walk_handles_parent_cycles() {
        let class_a = unique_class_name("runmat.unittest.CycleA");
        let class_b = unique_class_name("runmat.unittest.CycleB");

        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: class_a.clone(),
            parent: Some(class_b.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: class_b.clone(),
            parent: Some(class_a.clone()),
            properties: HashMap::new(),
            methods: HashMap::new(),
        });

        let obj = Value::Object(ObjectInstance::new(class_a.clone()));
        let not_found = isa_builtin(obj.clone(), Value::from("nonexistentType")).expect("isa");
        assert_eq!(not_found, Value::Bool(false));

        let parent_match = isa_builtin(obj, Value::from(class_b)).expect("isa");
        assert_eq!(parent_match, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_listener_alias_matches() {
        let listener = Listener {
            id: 1,
            target: test_handle_target(),
            target_class_name: "EventTarget".into(),
            event_name: "Changed".into(),
            callback: test_handle_target(),
            enabled: true,
            valid: true,
        };
        let value = Value::Listener(listener);
        assert_eq!(
            isa_builtin(value.clone(), Value::from("listener")).expect("isa"),
            Value::Bool(true)
        );
        assert_eq!(
            isa_builtin(value.clone(), Value::from("event.listener")).expect("isa"),
            Value::Bool(true)
        );
        assert_eq!(
            isa_builtin(value, Value::from("handle")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_meta_class_detection() {
        let meta = Value::ClassRef("Point".into());
        assert_eq!(
            isa_builtin(meta, Value::from("meta.class")).expect("isa"),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn isa_errors_on_invalid_type_designator() {
        let type_array = Value::StringArray(
            StringArray::new(vec!["double".into(), "single".into()], vec![1, 2]).unwrap(),
        );
        let err = isa_builtin(Value::Num(1.0), type_array).unwrap_err();
        let message = error_message(err);
        assert_eq!(
            message,
            "isa: TYPE must be a string scalar or character vector"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isa_gpuarray_with_wgpu_provider_matches_numeric_category() {
        use runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider;
        use runmat_accelerate_api::AccelProvider;

        let provider = match ensure_wgpu_provider() {
            Ok(Some(p)) => p,
            _ => return,
        };

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("wgpu upload");
        let value = Value::GpuTensor(handle);

        let numeric = isa_builtin(value.clone(), Value::from("numeric")).expect("isa numeric");
        assert_eq!(numeric, Value::Bool(true));

        let dbl = isa_builtin(value, Value::from("double")).expect("isa double");
        assert_eq!(dbl, Value::Bool(false));
    }
}
