use super::*;
use runmat_types::{CallableIdentity, MemberName, ProgramSourceId};
use runmat_vm::{BytecodeSourceRef, BytecodeSubscriptStep, FunctionBytecode, Instr};
use std::sync::Arc;

fn body(id: usize) -> FunctionBytecode {
    FunctionBytecode {
        function: FunctionId(id),
        portable_function: runmat_types::ProgramFunctionId(1),
        display_name: "stable".into(),
        source_id: Some(BytecodeSourceRef::Program(ProgramSourceId(0))),
        coverage_sites: vec![vec![41]],
        ..Default::default()
    }
}

#[test]
fn decorated_projection_proves_exact_group_without_changing_published_owner() {
    let prior = body(7);
    let owner = Arc::clone(&prior.definition_owner);
    let published = FunctionRegistry::new(HashMap::from([(FunctionId(7), prior)]));
    let incoming = body(70);
    let mut product = Bytecode::empty();
    product.function_registry = FunctionRegistry::new(HashMap::from([(FunctionId(70), incoming)]));
    let current = HashSet::from([FunctionId(70)]);
    let remap = unchanged_product_remap(&product, &current, &published, &HashSet::new())
        .expect("same decorated group");
    assert_eq!(remap, HashMap::from([(FunctionId(70), FunctionId(7))]));
    let selected = published.get_current(FunctionId(7)).unwrap();
    assert!(Arc::ptr_eq(&owner, &selected.definition_owner));
    assert_eq!(published.resolve_name("stable"), Some(FunctionId(7)));
}

#[test]
fn decoration_difference_rejects_reuse_without_changing_published_owner() {
    let prior = body(7);
    let owner = Arc::clone(&prior.definition_owner);
    let published = FunctionRegistry::new(HashMap::from([(FunctionId(7), prior)]));
    let mut incoming = body(70);
    incoming.coverage_sites = vec![vec![42]];
    let mut product = Bytecode::empty();
    product.function_registry = FunctionRegistry::new(HashMap::from([(FunctionId(70), incoming)]));
    assert!(unchanged_product_remap(
        &product,
        &HashSet::from([FunctionId(70)]),
        &published,
        &HashSet::new(),
    )
    .is_none());
    assert!(Arc::ptr_eq(
        &owner,
        &published
            .get_current(FunctionId(7))
            .unwrap()
            .definition_owner
    ));
}

#[test]
fn closed_external_and_dotted_references_prove_then_rebind_the_draft() {
    let external = |id| CallableIdentity::ExternalFunction {
        function: FunctionId(id),
        display_name: "stable".into(),
    };
    let dotted = |id| Instr::ReadSubscriptPath {
        steps: vec![BytecodeSubscriptStep::DottedInvoke {
            member: MemberName("stable".into()),
            arguments: vec![],
            semantic_fallback: Some(Arc::new(external(id))),
        }],
        selection: runmat_types::SequenceUse::RequireSingle,
        context: runmat_types::ObjectIndexingContext::Expression,
        to_sequence_register: false,
    };
    let mut prior = body(7);
    prior.coverage_sites.clear();
    prior.instructions = vec![
        Instr::CreateExternalBoundFunctionHandle(
            FunctionId(7),
            "stable".into(),
            runmat_types::FunctionSourceRole::Primary,
        ),
        dotted(7),
    ];
    let prior_owner = Arc::clone(&prior.definition_owner);
    let published = FunctionRegistry::new(HashMap::from([(FunctionId(7), prior)]));
    let mut current_body = body(70);
    current_body.coverage_sites.clear();
    current_body.instructions = vec![
        Instr::CreateExternalBoundFunctionHandle(
            FunctionId(70),
            "stable".into(),
            runmat_types::FunctionSourceRole::Primary,
        ),
        dotted(70),
    ];
    let source = crate::ExecutableSource::new("owner", "stable.m", "stable");
    let revision = crate::ExecutableRevision::derive(
        &source,
        None,
        crate::program_environment(crate::CompatMode::RunMat),
    );
    let source_map = crate::ExecutableSourceMap::new(vec![crate::SourceMapEntry {
        source_id: ProgramSourceId(0),
        owner_identity: source.owner_identity.clone(),
        relative_path: source.relative_path.clone(),
        content_revision: runmat_execution::Digest::sha256(source.text.as_bytes()),
        display_name: source.relative_path.clone(),
        full_path: None,
        text: source.text.clone(),
    }]);
    let mut product = Bytecode::empty();
    product.function_registry =
        FunctionRegistry::new(HashMap::from([(FunctionId(70), current_body)]));
    product.bound_functions = product.function_registry.functions.clone();
    product.instructions = vec![dotted(70)];
    product.layout = Some(runmat_vm::VmAssemblyLayout {
        functions: HashMap::new(),
        entrypoints: HashMap::new(),
        storage_bindings: HashMap::new(),
    });
    let mut draft = crate::executable::ExecutableUnitDraft::prepare(
        source,
        revision,
        source_map,
        runmat_mir::MirAssembly::default(),
        runmat_mir::analysis::AnalysisStore::default(),
        product,
    )
    .expect("decorate once");
    let remap = unchanged_product_remap(
        draft.bytecode(),
        &HashSet::from([FunctionId(70)]),
        &published,
        &HashSet::new(),
    )
    .expect("closed product external and dotted identities normalize");
    assert_eq!(remap, HashMap::from([(FunctionId(70), FunctionId(7))]));
    draft
        .rebind_current_definitions(&published, &remap)
        .expect("proved decorated draft rebinds");
    let unit = draft.seal();
    let body = unit.functions().get_current(FunctionId(7)).unwrap();
    assert!(Arc::ptr_eq(&prior_owner, &body.definition_owner));
    assert!(matches!(
        &body.instructions[0],
        Instr::CreateExternalBoundFunctionHandle(FunctionId(7), _, _)
    ));
    let Instr::ReadSubscriptPath { steps, .. } = &body.instructions[1] else {
        panic!("dotted identity retained");
    };
    assert!(matches!(
        &steps[0],
        BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } if matches!(identity.as_ref(), CallableIdentity::ExternalFunction {
            function: FunctionId(7),
            ..
        })
    ));
    assert!(matches!(
        &unit.bytecode().instructions[0],
        Instr::ReadSubscriptPath { steps, .. }
            if matches!(
                &steps[0],
                BytecodeSubscriptStep::DottedInvoke {
                    semantic_fallback: Some(identity),
                    ..
                } if matches!(identity.as_ref(), CallableIdentity::ExternalFunction {
                    function: FunctionId(7),
                    ..
                })
            )
    ));
}
