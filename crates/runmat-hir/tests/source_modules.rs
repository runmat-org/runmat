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
        &LoweringContext::empty().with_top_level_sources(&sources),
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
        &LoweringContext::empty().with_top_level_sources(&sources),
    )
    .err()
    .expect("mismatched source provenance must fail");
    assert!(error.to_string().contains("source provenance"));
}
