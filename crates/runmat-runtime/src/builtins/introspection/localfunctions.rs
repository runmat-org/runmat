use runmat_builtins::LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS;
use runmat_value::{CellArray, Value};

#[cfg(test)]
pub(crate) fn dispatch_localfunctions(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    dispatch_localfunctions_arguments(crate::call::FunctionArgumentView::from_slice(&args))
}

fn dispatch_localfunctions_arguments(
    args: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(crate::runtime_descriptor_error(
            "localfunctions",
            &LOCALFUNCTIONS_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let source_id = match crate::user_functions::current_active_semantic_function() {
        Some(function) => {
            crate::user_functions::source_function_for(function).map(|info| info.source_id)
        }
        None => crate::source_context::current_source_info().and_then(|info| info.source_id),
    };
    let Some(source_id) = source_id else {
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
        .map(crate::user_functions::bound_function_handle_from_source_info)
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
pub fn localfunctions_builtin_registered(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    dispatch_localfunctions_arguments(args.view())
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
        assert!(cell.data().is_empty());
    }

    #[test]
    fn localfunctions_returns_current_source_bound_handles() {
        let _source_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(3),
            "/tmp/localfunctions.m".to_string(),
            "handles = localfunctions();".to_string(),
        )])
        .expect("localfunctions source catalog admission");
        let _current_guard = crate::source_context::replace_current_source_id(Some(SourceId(99)));
        let _active =
            crate::user_functions::push_active_semantic_function(10).expect("call-port scope");
        let _catalog_guard =
            crate::user_functions::install_source_function_catalog(Some(Arc::new(vec![
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "main".into(),
                    function: 9,
                    role: runmat_types::FunctionSourceRole::Primary,
                    binding_origin: None,
                    source_file: None,
                    definition_owners: None,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "first".into(),
                    function: 10,
                    role: runmat_types::FunctionSourceRole::Local,
                    binding_origin: None,
                    source_file: None,
                    definition_owners: None,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "@anon0".into(),
                    function: 11,
                    role: runmat_types::FunctionSourceRole::Anonymous,
                    binding_origin: None,
                    source_file: None,
                    definition_owners: None,
                },
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(4),
                    name: "other".into(),
                    function: 12,
                    role: runmat_types::FunctionSourceRole::Local,
                    binding_origin: None,
                    source_file: None,
                    definition_owners: None,
                },
            ])))
            .expect("call-port scope");

        let value = dispatch_localfunctions(Vec::new()).expect("localfunctions succeeds");
        let Value::Cell(cell) = value else {
            panic!("expected cell array");
        };
        assert_eq!(cell.data().len(), 1);
        assert!(matches!(
            &cell.data()[0],
            Value::BoundFunctionHandle(handle)
                if handle.name == "first"
                    && handle.function == 10
                    && handle.source_role == runmat_types::FunctionSourceRole::Local
        ));
    }

    #[test]
    fn retained_execution_group_excludes_primary_and_new_published_replacement() {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let _entry = runtime.enter();
        let _source_guard = crate::source_context::replace_source_catalog(vec![(
            SourceId(3),
            "/old/file.m".to_string(),
            "handles = localfunctions();".to_string(),
        )])
        .expect("retained localfunctions source admission");
        let _current_guard = crate::source_context::replace_current_source_id(Some(SourceId(3)));
        let owner = Arc::new(runmat_value::FunctionDefinitionOwner::default());
        let owners = Arc::new(vec![(10, Arc::clone(&owner))]);
        let local = crate::user_functions::SourceFunctionInfo {
            source_id: SourceId(3),
            name: "old_local".into(),
            function: 10,
            role: runmat_types::FunctionSourceRole::Local,
            binding_origin: None,
            source_file: Some("/old/file.m".into()),
            definition_owners: Some(owners),
        };
        let primary = crate::user_functions::SourceFunctionInfo {
            source_id: SourceId(3),
            name: "old_primary".into(),
            function: 9,
            role: runmat_types::FunctionSourceRole::Primary,
            binding_origin: None,
            source_file: Some("/old/file.m".into()),
            definition_owners: None,
        };
        let _published =
            crate::user_functions::install_source_function_catalog(Some(Arc::new(vec![
                crate::user_functions::SourceFunctionInfo {
                    source_id: SourceId(3),
                    name: "replacement".into(),
                    function: 10,
                    role: runmat_types::FunctionSourceRole::Local,
                    binding_origin: None,
                    source_file: Some("/new/file.m".into()),
                    definition_owners: None,
                },
            ])))
            .expect("call-port scope");
        let selected_local = local.clone();
        let selected_group = vec![primary, local];
        let _selected = crate::user_functions::install_execution_source_resolver(Some(
            std::rc::Rc::new(crate::user_functions::ExecutionSourceFacts::new(
                std::rc::Rc::new(move |function| (function == 10).then(|| selected_local.clone())),
                std::rc::Rc::new(move |source| {
                    if source == SourceId(3) {
                        selected_group.clone()
                    } else {
                        Vec::new()
                    }
                }),
                std::rc::Rc::new(|_| None),
                std::rc::Rc::new(|_| None),
            )),
        ))
        .expect("call-port scope");
        let Value::Cell(cell) = dispatch_localfunctions(Vec::new()).expect("retained locals")
        else {
            panic!("cell result");
        };
        assert_eq!(cell.data().len(), 1);
        let Value::BoundFunctionHandle(handle) = &cell.data()[0] else {
            panic!("bound local");
        };
        assert_eq!(handle.name, "old_local");
        assert_eq!(handle.function, 10);
        assert_eq!(handle.source_file.as_deref(), Some("/old/file.m"));
        assert!(handle.authority.admits_local(&runtime.session_lease()));
    }
}
