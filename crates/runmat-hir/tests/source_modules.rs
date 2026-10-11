use runmat_hir::{lower, FunctionKind, LoweringContext, SourceUnitKind};
use runmat_types::ProgramSourceId;

#[test]
fn companion_functions_retain_their_source_module_and_primary_role() {
    let program = runmat_parser::parse(
        "value = helper(); function y = local(); y = 1; end; function y = helper(); y = 2; end",
    )
    .unwrap();
    assert_eq!(program.body.len(), 3);
    let sources = [ProgramSourceId(0), ProgramSourceId(0), ProgramSourceId(1)];
    let lowered = lower(
        &program,
        &LoweringContext::empty()
            .with_primary_source(ProgramSourceId(0))
            .with_top_level_sources(&sources),
    )
    .unwrap();
    let assembly = &lowered.assembly;
    assert_eq!(assembly.modules.len(), 2);
    assert_eq!(assembly.modules[0].source_id.0, 0);
    assert_eq!(assembly.modules[1].source_id.0, 1);
    assert!(matches!(
        assembly.modules[0].source_unit,
        SourceUnitKind::ScriptFile
    ));
    assert!(matches!(
        assembly.modules[1].source_unit,
        SourceUnitKind::FunctionFile
    ));
    assert!(assembly.modules[0].synthetic_entry_function.is_some());
    let helper = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "helper")
        .unwrap();
    let local = assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "local")
        .unwrap();
    assert!(matches!(helper.kind, FunctionKind::Named));
    assert_eq!(helper.module, assembly.modules[1].id);
    assert_eq!(local.module, assembly.modules[0].id);
    assert_eq!(
        assembly.function_source_role(helper),
        runmat_types::FunctionSourceRole::Primary
    );
    assert_eq!(
        assembly.function_source_role(local),
        runmat_types::FunctionSourceRole::Local
    );
}

#[test]
fn explicit_source_provenance_requires_one_source_per_statement() {
    let program = runmat_parser::parse("x = 1; y = 2;").unwrap();
    let sources = [ProgramSourceId(0)];
    let error = lower(
        &program,
        &LoweringContext::empty()
            .with_primary_source(ProgramSourceId(0))
            .with_top_level_sources(&sources),
    )
    .expect_err("mismatched source provenance must fail");
    assert!(error.to_string().contains("source provenance"));
}

#[test]
fn source_bearing_statements_require_an_explicit_primary_owner() {
    let program = runmat_parser::parse("x = 1;").unwrap();
    let sources = [ProgramSourceId(7)];
    let error = lower(
        &program,
        &LoweringContext::empty().with_top_level_sources(&sources),
    )
    .expect_err("statement ordinals cannot supply primary authority");
    assert!(error.to_string().contains("admitted primary source"));
}

#[test]
fn function_only_and_empty_programs_retain_nonzero_explicit_primary_source() {
    for source in [
        "function y = helper(); y = 1; end",
        "% no executable statements",
    ] {
        let program = runmat_parser::parse(source).unwrap();
        let sources = vec![ProgramSourceId(7); program.body.len()];
        let assembly = lower(
            &program,
            &LoweringContext::empty()
                .with_primary_source(ProgramSourceId(7))
                .with_top_level_sources(&sources),
        )
        .unwrap()
        .assembly;
        assert_eq!(assembly.primary_source, Some(ProgramSourceId(7)));
        assert!(assembly.entrypoints.is_empty());
        assert!(assembly
            .modules
            .iter()
            .any(|module| module.source_id == ProgramSourceId(7)));
        let mut encoded = serde_json::to_value(&assembly).unwrap();
        assert_eq!(encoded["primary_source"], serde_json::json!(7));
        encoded.as_object_mut().unwrap().remove("primary_source");
        let legacy: runmat_hir::HirAssembly = serde_json::from_value(encoded).unwrap();
        assert_eq!(
            legacy.primary_source, None,
            "old source-less HIR remains decodable"
        );
    }
}
