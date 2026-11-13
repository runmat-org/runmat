use super::*;

#[test]
fn nonzero_primary_table_does_not_infer_root_from_first_map_entry() {
    let primary = ProgramSourceId(7);
    let source = "function y = helper(); y = 1; end";
    let ast = runmat_parser::parse(source).unwrap();
    let sources = [primary];
    let frontend = runmat_static_analysis::frontend::analyze_program_with_catalog(
        &ast,
        &runmat_hir::LoweringContext::empty()
            .with_primary_source(primary)
            .with_top_level_sources(&sources),
        None,
    );
    let hir = &frontend.lowering.as_ref().expect("HIR").assembly;
    let bytecode = frontend.bytecode.expect("function-only product");
    let entries = (0..=primary.0)
        .map(|ordinal| {
            let text = format!("source {ordinal}");
            crate::SourceMapEntry {
                source_id: ProgramSourceId(ordinal),
                owner_identity: "fixture".into(),
                relative_path: format!("source-{ordinal}.m"),
                content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
                display_name: format!("source-{ordinal}.m"),
                full_path: None,
                text,
            }
        })
        .collect();
    let table = ProgramSourceTable {
        primary_source: primary,
        source_map: crate::ExecutableSourceMap::new(entries),
        statement_sources: sources.to_vec(),
        function_sources: HashMap::new(),
        retained_runtime_sources: HashMap::new(),
    };
    table
        .validate_compiled_sources(&bytecode, hir)
        .expect("typed primary is independent of first map record");
}

#[test]
fn admitted_function_source_must_match_its_hir_module_not_merely_any_map_entry() {
    let source = "function y = primary(); y = 1; end\nfunction y = companion(); y = 2; end";
    let ast = runmat_parser::parse(source).expect("two functions");
    assert_eq!(ast.body.len(), 2);
    let sources = [ProgramSourceId(0), ProgramSourceId(1)];
    let frontend = runmat_static_analysis::frontend::analyze_program_with_catalog(
        &ast,
        &runmat_hir::LoweringContext::empty()
            .with_primary_source(ProgramSourceId(0))
            .with_top_level_sources(&sources),
        None,
    );
    assert!(frontend.compile_failure.is_none());
    let hir = &frontend.lowering.as_ref().expect("HIR").assembly;
    let mut bytecode = frontend.bytecode.expect("compiled function file");
    let record = |source_id, relative_path: &str, text: &str| crate::SourceMapEntry {
        source_id,
        owner_identity: "fixture".into(),
        relative_path: relative_path.into(),
        content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
        display_name: relative_path.into(),
        full_path: None,
        text: text.into(),
    };
    let table = ProgramSourceTable {
        primary_source: ProgramSourceId(0),
        source_map: crate::ExecutableSourceMap::new(vec![
            record(
                ProgramSourceId(0),
                "primary.m",
                "function y = primary(); y = 1; end",
            ),
            record(
                ProgramSourceId(1),
                "companion.m",
                "function y = companion(); y = 2; end",
            ),
        ]),
        statement_sources: sources.to_vec(),
        function_sources: HashMap::new(),
        retained_runtime_sources: HashMap::new(),
    };
    table
        .validate_compiled_sources(&bytecode, hir)
        .expect("exact HIR sources");
    let companion = hir
        .functions
        .iter()
        .find(|function| function.name.0 == "companion")
        .expect("companion HIR function");
    bytecode
        .function_registry
        .functions
        .get_mut(&companion.id)
        .unwrap()
        .source_id = Some(runmat_vm::BytecodeSourceRef::Program(ProgramSourceId(0)));
    assert!(table.validate_compiled_sources(&bytecode, hir).is_err());
}

#[test]
fn overlapping_companion_diagnostics_keep_exact_source_owner() {
    let primary = "function C = entry_a(); A=rand(3,4); B=rand(3,4); C=A*B; end";
    let companion = "function C = entry_b(); A=rand(3,4); B=rand(3,4); C=A*B; end";
    let mut ast = runmat_parser::parse(primary).expect("primary source");
    ast.body.extend(
        runmat_parser::parse(companion)
            .expect("companion source")
            .body,
    );
    let sources = [ProgramSourceId(0), ProgramSourceId(1)];
    let frontend = runmat_static_analysis::frontend::analyze_program_with_catalog(
        &ast,
        &runmat_hir::LoweringContext::empty()
            .with_primary_source(sources[0])
            .with_top_level_sources(&sources),
        None,
    );
    let diagnostics = frontend
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "RM-TYPE-MATMUL")
        .collect::<Vec<_>>();
    assert_eq!(
        diagnostics.len(),
        2,
        "same local offsets must not deduplicate distinct sources"
    );
    assert_eq!(diagnostics[0].primary.span, diagnostics[1].primary.span);
    assert_eq!(diagnostics[0].primary.source, Some(sources[0]));
    assert_eq!(diagnostics[1].primary.source, Some(sources[1]));
    let record = |source_id, name: &str, text: &str| crate::SourceMapEntry {
        source_id,
        owner_identity: "fixture".into(),
        relative_path: name.into(),
        content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
        display_name: name.into(),
        full_path: None,
        text: text.into(),
    };
    let table = ProgramSourceTable {
        primary_source: sources[0],
        source_map: crate::ExecutableSourceMap::new(vec![
            record(sources[0], "entry_a.m", primary),
            record(sources[1], "entry_b.m", companion),
        ]),
        statement_sources: sources.to_vec(),
        function_sources: HashMap::new(),
        retained_runtime_sources: HashMap::new(),
    };
    let error = super::super::bind_frontend_diagnostic(diagnostics[1], &table);
    let crate::RunError::SourceCompile(bound) = error else {
        panic!("companion diagnostic must retain its source context");
    };
    assert_eq!(bound.source.source_id, sources[1]);
    assert_eq!(bound.source.display_name, "entry_b.m");
    assert_eq!(bound.source.text, companion);
    assert_eq!(bound.error.identifier.as_deref(), Some("RM-TYPE-MATMUL"));

    let vm_error = runmat_vm::CompileError::new("companion VM compilation failed")
        .with_identifier("RunMat:VmCompileFixture")
        .with_span(diagnostics[1].primary.span)
        .with_source(sources[1]);
    let crate::RunError::SourceCompile(bound) = super::super::bind_compile_error(vm_error, &table)
    else {
        panic!("early VM compile failures must bind their exact source");
    };
    assert_eq!(bound.source.source_id, sources[1]);
    assert_eq!(
        bound.error.identifier.as_deref(),
        Some("RunMat:VmCompileFixture")
    );
    assert_eq!(bound.error.span, Some(diagnostics[1].primary.span));
}

#[test]
fn reverse_projection_uses_issued_authority_not_pool_integer() {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let _unrelated = runmat_runtime::source_context::lease_source_context_in_context(
        &runtime,
        "unrelated.m",
        None,
        "unrelated = 1;",
    )
    .expect("advance the selected source issuer");
    let text = "function y = main(); y = 2; end";
    let table = ProgramSourceTable {
        primary_source: ProgramSourceId(0),
        source_map: crate::ExecutableSourceMap::new(vec![crate::SourceMapEntry {
            source_id: ProgramSourceId(0),
            owner_identity: "fixture".into(),
            relative_path: "main.m".into(),
            content_revision: runmat_execution::Digest::sha256(text.as_bytes()),
            display_name: "main.m".into(),
            full_path: None,
            text: text.into(),
        }]),
        statement_sources: Vec::new(),
        function_sources: HashMap::new(),
        retained_runtime_sources: HashMap::new(),
    };
    let mut pool = crate::source_pool::SourcePool::default();
    let projection = table
        .intern_runtime_projection(&mut pool, &runtime)
        .expect("issued projection");
    let issued = projection.program_to_runtime[&ProgramSourceId(0)];
    assert_eq!(pool.entries().next().map(|(id, _)| id.0), Some(0));
    assert_eq!(issued.source_id().0, 1);
    assert_eq!(
        projection.program_ref_for_runtime_ref(runmat_vm::BytecodeSourceRef::Runtime(issued)),
        Some(runmat_vm::BytecodeSourceRef::Program(ProgramSourceId(0)))
    );
}
