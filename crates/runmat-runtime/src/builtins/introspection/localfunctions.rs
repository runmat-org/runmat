use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamDescriptor, BuiltinSignatureDescriptor, CellArray, Value,
};

const LOCALFUNCTIONS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "handles",
    ty: runmat_builtins::BuiltinParamType::Any,
    arity: runmat_builtins::BuiltinParamArity::Required,
    default: None,
    description: "Cell array of visible local function handles.",
}];

const LOCALFUNCTIONS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "handles = localfunctions()",
    inputs: &[],
    outputs: &LOCALFUNCTIONS_OUTPUT,
}];

pub const LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOCALFUNCTIONS.TOO_MANY_INPUTS",
    identifier: Some("RunMat:TooManyInputs"),
    when: "Any input argument is provided.",
    message: "localfunctions: too many input arguments",
};

pub const LOCALFUNCTIONS_ERRORS: [BuiltinErrorDescriptor; 1] =
    [LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS];

pub const LOCALFUNCTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOCALFUNCTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOCALFUNCTIONS_ERRORS,
};

fn is_supported_local_function_name(name: &str) -> bool {
    let trimmed = name.trim();
    !trimmed.is_empty()
        && !trimmed.starts_with("@anon")
        && !trimmed.starts_with("anonymous#")
        && !trimmed.contains('.')
}

pub(crate) fn dispatch_localfunctions(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "localfunctions",
            &LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let Some(source_id) =
        crate::source_context::current_source_info().and_then(|info| info.source_id)
    else {
        return Ok(Value::Cell(CellArray::new(Vec::new(), 1, 0).map_err(
            |err| {
                crate::build_runtime_error(format!("localfunctions: {err}"))
                    .with_builtin("localfunctions")
                    .build()
            },
        )?));
    };
    let handles = crate::user_functions::source_functions_for(source_id)
        .into_iter()
        .filter(|info| is_supported_local_function_name(&info.name))
        .map(|info| Value::BoundFunctionHandle {
            name: info.name,
            function: info.function,
        })
        .collect::<Vec<_>>();
    let len = handles.len();
    Ok(Value::Cell(CellArray::new(handles, 1, len).map_err(
        |err| {
            crate::build_runtime_error(format!("localfunctions: {err}"))
                .with_builtin("localfunctions")
                .build()
        },
    )?))
}

#[runmat_macros::runtime_builtin(
    name = "localfunctions",
    category = "introspection",
    summary = "Return handles for named local functions visible in the current source.",
    descriptor(self::LOCALFUNCTIONS_DESCRIPTOR),
    builtin_path = "crate::builtins::introspection::localfunctions"
)]
pub fn localfunctions_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_localfunctions(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_hir::SourceId;
    use std::sync::Arc;

    #[test]
    fn localfunctions_returns_current_source_bound_handles() {
        let _source_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(3),
            "/tmp/localfunctions.m".to_string(),
            "handles = localfunctions();".to_string(),
        )]);
        let _current_guard = crate::source_context::replace_current_source_id(Some(SourceId(3)));
        let _catalog_guard =
            crate::user_functions::install_source_function_catalog(Some(Arc::new(vec![
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "first".to_string(),
                    function: 10,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "@anon0".to_string(),
                    function: 11,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(4),
                    name: "other".to_string(),
                    function: 12,
                },
            ])));

        let value = dispatch_localfunctions(Vec::new()).expect("localfunctions succeeds");
        let Value::Cell(cell) = value else {
            panic!("expected cell array");
        };
        assert_eq!(cell.data.len(), 1);
        assert!(matches!(
            &*cell.data[0],
            Value::BoundFunctionHandle { name, function } if name == "first" && *function == 10
        ));
    }
}
