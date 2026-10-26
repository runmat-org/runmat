use super::invoke_semantic_function_value_in_context;
use crate::bytecode::{
    BytecodeSourceRef, FunctionBytecode, FunctionRegistry, Instr, PrivateSourceDirectory,
};
use runmat_hir::FunctionId;
use runmat_value::Value;
use std::{collections::HashMap, rc::Rc, sync::Arc};

fn caller(source: &str, owner: &str) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(1),
        display_name: "entry".into(),
        source_file: Some(source.into()),
        private_source_owner: Some(PrivateSourceDirectory::new(owner.into())),
        instructions: vec![
            Instr::LoadString("helper".into()),
            Instr::CallFevalMulti(0, 1),
            Instr::StoreVar(0),
            Instr::Return,
        ],
        var_count: 1,
        output_slots: vec![0],
        ..Default::default()
    }
}

fn helper(source: &str, owner: &str, result: f64) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(2),
        display_name: "__private__.helper".into(),
        source_file: Some(source.into()),
        private_source_owner: Some(PrivateSourceDirectory::new(owner.into())),
        instructions: vec![Instr::LoadConst(result), Instr::StoreVar(0), Instr::Return],
        var_count: 1,
        output_slots: vec![0],
        ..Default::default()
    }
}

#[test]
fn retained_private_name_executes_old_body_after_same_id_republication() {
    runmat_gc::gc_test_context(|| {
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let old_caller = caller("old/entry.m", "old");
        let old_source = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime,
            "old/private/helper.m",
            None,
            "old private source",
        )
        .expect("old private source authority");
        let new_source = runmat_runtime::source_context::lease_source_context_in_context(
            &runtime,
            "new/private/helper.m",
            None,
            "new private source",
        )
        .expect("current private source authority");
        let mut old_helper = helper("old/private/helper.m", "old", 33.0);
        old_helper.source_id = Some(BytecodeSourceRef::Runtime(old_source.source_ref()));
        let owners = Arc::new(vec![
            (1, Arc::clone(&old_caller.definition_owner)),
            (2, Arc::clone(&old_helper.definition_owner)),
        ]);
        let authority = {
            let _entry = runtime.enter();
            let _catalog = runmat_runtime::user_functions::install_source_function_catalog(Some(
                Arc::new(vec![runmat_runtime::user_functions::SourceFunctionInfo {
                    source_id: runmat_types::SourceId(0),
                    name: "entry".into(),
                    function: 1,
                    role: runmat_types::FunctionSourceRole::Primary,
                    binding_origin: None,
                    source_file: Some("old/entry.m".into()),
                    definition_owners: Some(owners),
                }]),
            ))
            .expect("call-port scope");
            runmat_runtime::user_functions::bound_function_authority_for(1)
        };
        // A published ordinary helper must not eclipse the caller-private
        // selection, including an old caller whose private body is leased.
        let mut public_helper = helper("helper.m", "", 99.0);
        public_helper.function = FunctionId(3);
        public_helper.display_name = "helper".into();
        let mut registry = FunctionRegistry::new(HashMap::from([
            (FunctionId(1), old_caller),
            (FunctionId(2), old_helper),
            (FunctionId(3), public_helper),
        ]));
        registry.insert_replacing_name(caller("new/entry.m", "new"));
        let mut new_helper = helper("new/private/helper.m", "new", 22.0);
        new_helper.source_id = Some(BytecodeSourceRef::Runtime(new_source.source_ref()));
        registry.insert_replacing_name(new_helper);
        let retained = registry
            .owned_view(FunctionId(1), &authority)
            .expect("old caller and helper remain leased");
        assert_eq!(
            retained.resolve_private_name_for_caller(FunctionId(1), "helper"),
            Some(FunctionId(2))
        );
        let selected = retained
            .semantic_name_candidate_facts_in(&runtime, [FunctionId(2)])
            .remove(&FunctionId(2))
            .expect("retained private fact");
        let published = registry
            .current_view()
            .semantic_name_candidate_facts_in(&runtime, [FunctionId(2)])
            .remove(&FunctionId(2))
            .expect("current publication fact");
        assert!(!selected.same_definition(&published));
        assert_eq!(
            selected
                .source
                .as_ref()
                .and_then(|info| info.source_file.as_deref()),
            Some("old/private/helper.m")
        );
        assert_eq!(
            published
                .source
                .as_ref()
                .and_then(|info| info.source_file.as_deref()),
            Some("new/private/helper.m")
        );
        let old = futures::executor::block_on(invoke_semantic_function_value_in_context(
            1,
            &[],
            1,
            &retained,
            runtime.clone(),
        ))
        .expect("leased caller executes private name");
        assert_eq!(old.as_slice(), &[Value::Num(33.0)]);
        let fresh_caller = futures::executor::block_on(invoke_semantic_function_value_in_context(
            1,
            &[],
            1,
            &registry.current_view(),
            runtime.clone(),
        ))
        .expect("current caller resolves its private helper before the ordinary name");
        assert_eq!(fresh_caller.as_slice(), &[Value::Num(22.0)]);
        let current = futures::executor::block_on(invoke_semantic_function_value_in_context(
            2,
            &[],
            1,
            &registry.current_view(),
            runtime,
        ))
        .expect("numeric current body remains published");
        assert_eq!(current.as_slice(), &[Value::Num(22.0)]);
    });
}
