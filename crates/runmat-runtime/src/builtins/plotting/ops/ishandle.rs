use crate::builtins::plotting::type_resolvers::handle_logical_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerAuditDescriptor, BuiltinIntegerAuditKind,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_builtins::{LogicalArray, Value};
use runmat_macros::runtime_builtin;

const ISHANDLE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "True when input resolves to a valid plotting handle.",
}];

const ISHANDLE_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Candidate plotting handle value.",
}];

const ISHANDLE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = ishandle(h)",
    inputs: &ISHANDLE_INPUTS_HANDLE,
    outputs: &ISHANDLE_OUTPUT,
}];

const ISHANDLE_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISHANDLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISHANDLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISHANDLE_ERRORS,
};

const ISHANDLE_INTEGER_HANDLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ishandle-integer-handle-alias",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ishandle numeric integer handle aliases are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:IshandleIntegerHandleExtension"),
};
const ISHANDLE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [ISHANDLE_INTEGER_HANDLE_EXTENSION];
pub const ISHANDLE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "MATLAB graphics predicates operate on graphics objects; numeric integer handle aliases are a separately declared RunMat-only extension and compatibility mode returns same-shaped false.",
};

#[runtime_builtin(
    name = "ishandle",
    category = "plotting",
    summary = "Return true if the input is a valid plotting handle.",
    keywords = "ishandle,plotting,handle",
    suppress_auto_output = true,
    type_resolver(handle_logical_type),
    descriptor(crate::builtins::plotting::ishandle::ISHANDLE_DESCRIPTOR),
    extensions(ISHANDLE_EXTENSIONS),
    integer_audit(crate::builtins::plotting::ishandle::ISHANDLE_INTEGER_AUDIT),
    builtin_path = "crate::builtins::plotting::ishandle"
)]
pub fn ishandle_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.len() != 1 {
        return Err(crate::builtins::plotting::plotting_error(
            "ishandle",
            "ishandle: expected exactly one input argument",
        ));
    }
    Ok(handle_predicate_value(&args[0], "ishandle"))
}

fn handle_predicate_value(value: &Value, builtin: &'static str) -> Value {
    match value {
        Value::GpuTensor(handle) => false_value(handle.shape.clone()),
        Value::Int(integer) => {
            if !crate::compatibility::runmat_extensions_enabled() {
                return Value::Bool(false);
            }
            Value::Bool(
                crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(integer)
                    .is_some_and(|handle| handle_is_valid(handle, builtin)),
            )
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if crate::compatibility::ensure_builtin_extension_enabled(
                &ISHANDLE_INTEGER_HANDLE_EXTENSION,
                builtin,
            )
            .is_err()
            {
                return Value::LogicalArray(
                    LogicalArray::new(vec![0; tensor.len()], tensor.shape.clone())
                        .expect("logical shape from tensor"),
                );
            }
            integer_handle_predicate(tensor, builtin, handle_is_valid)
        }
        Value::Tensor(tensor) => numeric_handle_predicate(tensor, builtin, handle_is_valid),
        _ => Value::Bool(
            crate::builtins::plotting::properties::resolve_plot_handle(value, builtin).is_ok(),
        ),
    }
}

fn false_value(shape: Vec<usize>) -> Value {
    if shape.iter().product::<usize>() == 1 {
        Value::Bool(false)
    } else {
        Value::LogicalArray(
            LogicalArray::new(vec![0; shape.iter().product()], shape)
                .expect("logical shape from input metadata"),
        )
    }
}

fn integer_handle_predicate(
    tensor: &runmat_builtins::Tensor,
    builtin: &'static str,
    predicate: fn(f64, &'static str) -> bool,
) -> Value {
    let storage = tensor
        .integer_storage()
        .expect("integer handle predicate requires integer storage");
    let data = storage
        .exact_values()
        .iter()
        .map(|value| {
            u8::from(
                crate::builtins::plotting::op_common::handles::numeric_handle_from_integer(value)
                    .is_some_and(|handle| predicate(handle, builtin)),
            )
        })
        .collect();
    Value::LogicalArray(
        LogicalArray::new(data, tensor.shape.clone()).expect("logical shape from tensor"),
    )
}

fn numeric_handle_predicate(
    tensor: &runmat_builtins::Tensor,
    builtin: &'static str,
    predicate: fn(f64, &'static str) -> bool,
) -> Value {
    let data = tensor
        .materialize_f64()
        .iter()
        .map(|&handle| u8::from(predicate(handle, builtin)))
        .collect();
    Value::LogicalArray(
        LogicalArray::new(data, tensor.shape.clone()).expect("logical shape from tensor"),
    )
}

fn handle_is_valid(handle: f64, builtin: &'static str) -> bool {
    if !handle.is_finite() || handle < 0.0 {
        return false;
    }
    crate::builtins::plotting::properties::resolve_plot_handle(&Value::Num(handle), builtin).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ishandle_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ISHANDLE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"tf = ishandle(h)"));
    }

    #[test]
    fn ishandle_vectorizes_numeric_handle_arrays() {
        let handles = runmat_builtins::Tensor::new(vec![f64::NAN, -1.0], vec![2, 1]).unwrap();
        let result = ishandle_builtin(vec![Value::Tensor(handles)]).unwrap();
        match result {
            Value::LogicalArray(logical) => {
                assert_eq!(logical.shape, vec![2, 1]);
                assert_eq!(logical.data, vec![0, 0]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn ishandle_vectorizes_typed_handles_without_a_floating_mirror() {
        let handles = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::I16(vec![-1, -2]),
            vec![2, 1],
        )
        .unwrap();
        let result = ishandle_builtin(vec![Value::Tensor(handles)]).unwrap();
        match result {
            Value::LogicalArray(logical) => assert_eq!(logical.data, vec![0, 0]),
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn compatibility_mode_treats_integer_arrays_as_nonobjects() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let handles = runmat_builtins::Tensor::new_integer(
            runmat_builtins::IntegerStorage::U64(vec![u64::MAX]),
            vec![1, 1],
        )
        .unwrap();
        let Value::LogicalArray(result) = ishandle_builtin(vec![Value::Tensor(handles)]).unwrap()
        else {
            panic!("expected logical array");
        };
        assert_eq!(result.data, vec![0]);
    }

    #[test]
    fn ishandle_requires_exactly_one_argument() {
        assert!(ishandle_builtin(Vec::new()).is_err());
        assert!(ishandle_builtin(vec![Value::Num(0.0), Value::Num(1.0)]).is_err());
        assert_eq!(
            ishandle_builtin(vec![Value::Num(0.0)]).unwrap(),
            Value::Bool(true)
        );
    }
}
