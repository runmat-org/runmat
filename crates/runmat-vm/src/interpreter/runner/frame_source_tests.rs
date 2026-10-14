use super::*;
use std::{collections::HashMap, rc::Rc};

use runmat_execution::{Digest, ExecutableSourceMapV2, ExecutableSourceRecordV2};
use runmat_types::{ProgramFunctionId, ProgramSourceId};

fn selected_runtime() -> runmat_runtime::context::RuntimeContext {
    let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let text = "value = 1;";
    let map = ExecutableSourceMapV2::new(vec![ExecutableSourceRecordV2 {
        source_id: ProgramSourceId(0),
        owner_identity: "fixture".into(),
        relative_path: "selected.m".into(),
        content_revision: Digest::sha256(text.as_bytes()),
        display_name: "selected.m".into(),
        text: text.into(),
    }])
    .expect("valid portable map");
    runmat_runtime::source_context::bind_portable_source_map_v2(runtime, &map)
        .expect("one selected source issuance")
}

#[test]
fn call_frame_basis_uses_selected_body_or_exact_script_table() {
    let runtime = selected_runtime();
    let issued = runtime
        .program_source_ref(ProgramSourceId(0))
        .expect("issued source");
    let mut script = Bytecode::with_instructions(Vec::new(), 0);
    script.source_id = Some(crate::BytecodeSourceRef::Runtime(issued));
    let expected = runtime
        .call_frame_source_owner_for_program_source(ProgramSourceId(0))
        .expect("selected table basis");
    let (script_basis, script_diagnostic) =
        selected_frame_source_basis(&script, &runtime).expect("selected script basis");
    assert_eq!(script_basis, expected);
    assert_eq!(script_diagnostic.display_name(), "selected.m");
    assert_eq!(script_diagnostic.line_column(Some((8, 8))), Some((1, 9)));

    let portable = runtime
        .portable_program_source_map()
        .expect("selected portable map");
    let owner =
        runmat_runtime::source_context::SelectedProgramSourceOwner::capture(&runtime, portable)
            .expect("bound owner");
    let program = Digest::sha256("selected program");
    let function = runmat_hir::FunctionId(7);
    let mut body = crate::FunctionBytecode {
        function,
        source_id: script.source_id,
        binding_origin: Some(runmat_execution::FunctionBindingOrigin {
            program,
            function: ProgramFunctionId(7),
        }),
        ..crate::FunctionBytecode::default()
    };
    body.display_name = "selected".into();
    let mut registry = crate::FunctionRegistry::new(HashMap::from([(function, body)]));
    registry
        .attach_selected_program_source_owner([function], program, &owner)
        .expect("exact function/source admission");
    script.active_function = Some(function);
    script.function_registry = registry;
    let (function_basis, function_diagnostic) =
        selected_frame_source_basis(&script, &runtime).expect("selected function basis");
    assert_eq!(function_basis, expected);
    assert_eq!(function_diagnostic.display_name(), "selected.m");
    assert_eq!(function_diagnostic.line_column(Some((8, 8))), Some((1, 9)));

    let rebound =
        runmat_runtime::source_context::bind_portable_source_map_v2(runtime.clone(), portable)
            .expect("same session may admit a separate identical source table");
    let new_basis = rebound
        .call_frame_source_owner_for_program_source(ProgramSourceId(0))
        .expect("new table basis");
    assert_ne!(expected, new_basis);
    let (rebound_basis, rebound_diagnostic) =
        selected_frame_source_basis(&script, &rebound).expect("retained function basis");
    assert_eq!(
        rebound_basis, expected,
        "a retained function keeps its old exact table through a later admission"
    );
    assert_eq!(rebound_diagnostic.display_name(), "selected.m");
    assert_eq!(rebound_diagnostic.line_column(Some((8, 8))), Some((1, 9)));

    script.source_id = None;
    assert!(selected_frame_source_basis(&script, &runtime).is_none());
    script.source_id = Some(crate::BytecodeSourceRef::Runtime(issued));
    script.active_function = Some(runmat_hir::FunctionId(8));
    assert!(selected_frame_source_basis(&script, &runtime).is_none());
}
