use runmat_builtins::LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS;
use runmat_value::{CellArray, Value};

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
        .filter(|info| info.role == runmat_types::FunctionSourceRole::Local)
        .map(|info| Value::BoundFunctionHandle {
            name: info.name,
            function: info.function,
            source_role: info.role,
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
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::localfunctions"
)]
pub fn localfunctions_builtin_registered(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_localfunctions(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::SourceId;
    use std::sync::Arc;

    #[test]
    fn localfunctions_without_source_is_exact_empty_cell_row() {
        let _current_guard = crate::source_context::replace_current_source_id(None);
        let value = dispatch_localfunctions(Vec::new()).expect("empty source result");
        let Value::Cell(cell) = value else {
            panic!("expected cell array");
        };
        assert_eq!(cell.rows, 1);
        assert_eq!(cell.cols, 0);
        assert!(cell.data.is_empty());
    }

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
                    name: "main".into(),
                    function: 9,
                    role: runmat_types::FunctionSourceRole::Primary,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "first".into(),
                    function: 10,
                    role: runmat_types::FunctionSourceRole::Local,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "@anon0".into(),
                    function: 11,
                    role: runmat_types::FunctionSourceRole::Anonymous,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(4),
                    name: "other".into(),
                    function: 12,
                    role: runmat_types::FunctionSourceRole::Local,
                },
            ])));

        let value = dispatch_localfunctions(Vec::new()).expect("localfunctions succeeds");
        let Value::Cell(cell) = value else {
            panic!("expected cell array");
        };
        assert_eq!(cell.data.len(), 1);
        assert!(matches!(
            &cell.data[0],
            Value::BoundFunctionHandle { name, function, source_role }
                if name == "first"
                    && *function == 10
                    && *source_role == runmat_types::FunctionSourceRole::Local
        ));
    }
}
