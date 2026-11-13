use runmat_hir::{lower, LoweringContext};
use runmat_mir::{analysis::analyze_assembly, lowering::lower_assembly};
use runmat_types::{ProgramFunctionId, ProgramSourceId};

#[test]
fn companion_function_and_entrypoint_have_distinct_mir_source_ordinals() {
    let program = runmat_parser::parse(
        "value = companion(); function y = local(); y = 1; end; function y = companion(); y = 2; end",
    )
    .unwrap();
    let sources = [ProgramSourceId(0), ProgramSourceId(0), ProgramSourceId(3)];
    let hir = lower(
        &program,
        &LoweringContext::empty()
            .with_primary_source(ProgramSourceId(0))
            .with_top_level_sources(&sources),
    )
    .unwrap();
    let mir = lower_assembly(&hir.assembly).unwrap();
    let entry = mir.entrypoints[0];
    assert_eq!(mir.functions[&entry].source, ProgramSourceId(0));
    let companion = hir
        .assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "companion")
        .unwrap();
    let local = hir
        .assembly
        .functions
        .iter()
        .find(|function| function.name.0 == "local")
        .unwrap();
    assert_eq!(mir.functions[&companion.id].source, ProgramSourceId(3));
    assert_eq!(mir.functions[&local.id].source, ProgramSourceId(0));
    let facts = analyze_assembly(&mir);
    assert!(facts.local_value_count(Some(ProgramFunctionId(companion.id.0 as u32))) > 0);
}
