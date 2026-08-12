use crate::builtins::plotting::type_resolvers::handle_logical_type;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
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

const ISHANDLE_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];

const ISHANDLE_INPUTS_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Candidate plotting handle value.",
}];

const ISHANDLE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "tf = ishandle()",
        inputs: &ISHANDLE_INPUTS_NONE,
        outputs: &ISHANDLE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = ishandle(h)",
        inputs: &ISHANDLE_INPUTS_HANDLE,
        outputs: &ISHANDLE_OUTPUT,
    },
];

const ISHANDLE_ERRORS: [BuiltinErrorDescriptor; 0] = [];

pub const ISHANDLE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISHANDLE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ISHANDLE_ERRORS,
};

#[runtime_builtin(
    name = "ishandle",
    category = "plotting",
    summary = "Return true if the input is a valid plotting handle.",
    keywords = "ishandle,plotting,handle",
    suppress_auto_output = true,
    type_resolver(handle_logical_type),
    descriptor(crate::builtins::plotting::ishandle::ISHANDLE_DESCRIPTOR),
    builtin_path = "crate::builtins::plotting::ishandle"
)]
pub fn ishandle_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let Some(value) = args.first() else {
        return Ok(Value::Bool(false));
    };
    Ok(handle_predicate_value(value, "ishandle"))
}

fn handle_predicate_value(value: &Value, builtin: &'static str) -> Value {
    match value {
        Value::Tensor(tensor) => {
            let data = tensor
                .materialize_f64()
                .iter()
                .map(|&handle| u8::from(handle_is_valid(handle, builtin)))
                .collect();
            Value::LogicalArray(
                LogicalArray::new(data, tensor.shape.clone()).expect("logical shape from tensor"),
            )
        }
        _ => Value::Bool(
            crate::builtins::plotting::properties::resolve_plot_handle(value, builtin).is_ok(),
        ),
    }
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
        assert!(labels.contains(&"tf = ishandle()"));
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
}
