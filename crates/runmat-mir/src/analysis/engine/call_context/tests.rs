use super::*;
use runmat_types::ValueKindFact;

fn lower(source: &str) -> MirAssembly {
    let ast = runmat_parser::parse(source).expect("parse call-context fixture");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
        .expect("lower call-context HIR");
    let mut mir = crate::lowering::lower_assembly(&hir.assembly).expect("lower call-context MIR");
    // Distinct source ownership is transported by MIR metadata. The shared
    // project-frontend tests additionally qualify real separately parsed files.
    for metadata in mir.functions.values_mut() {
        if metadata.name == runmat_hir::FunctionName("helper".into()) {
            metadata.source = ProgramSourceId(1);
        }
    }
    mir
}

const HELPER: &str = "\nfunction C = helper(A, B)\nC = A * B;\nend\n";

#[test]
fn actual_ones_producers_supply_exact_call_facts_and_local_matrix_error() {
    let local = lower("A = ones(2,3); B = ones(4,2); C = A * B;");
    assert!(crate::analysis::analyze_assembly(&local)
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-MATMUL"));

    let mir = lower(&format!(
        "A = ones(2,3); B = ones(4,2); C = helper(A,B);{HELPER}"
    ));
    let generic = interprocedural::analyze_program(&mir);
    let (driver, metadata) = mir
        .functions
        .iter()
        .find(|(_, metadata)| metadata.kind == runmat_hir::FunctionKind::SyntheticEntrypoint)
        .expect("driver entry");
    assert_eq!(metadata.source, ProgramSourceId(0));
    let body = &mir.bodies[driver];
    let result = flow::analyze_body(
        body,
        ProgramFunctionId(u32::try_from(driver.0).unwrap()),
        generic.parameters.get(driver).map_or(&[], Vec::as_slice),
        generic.captures.get(driver).map_or(&[], Vec::as_slice),
        None,
        &generic.summaries,
    );
    let call = result
        .calls
        .iter()
        .find(|call| mir.functions[&call.callee].name == runmat_hir::FunctionName("helper".into()))
        .expect("real caller observation");
    assert!(call.complete_arguments);
    assert_eq!(call.arguments.len(), 2);
    assert_eq!(
        call.arguments[0].shape.known_dims(),
        Some(vec![Some(2), Some(3)])
    );
    assert_eq!(
        call.arguments[1].shape.known_dims(),
        Some(vec![Some(4), Some(2)])
    );
}

#[test]
fn incompatible_call_uses_typed_operator_diagnostic_without_narrowing_public_facts() {
    let source = format!("A = ones(2,3); B = ones(4,2); C = helper(A,B);{HELPER}");
    let mir = lower(&source);
    let before = crate::analysis::analyze_assembly(&mir);
    let report = diagnose_call_contexts(&mir);
    assert!(!report.budget_exhausted);
    assert!(!report.precision_withheld);
    let diagnostics = report.diagnostics;
    let [diagnostic] = diagnostics.as_slice() else {
        panic!("one contextual matrix failure: {diagnostics:#?}");
    };
    assert_eq!(diagnostic.code, "RM-TYPE-MATMUL");
    assert_eq!(diagnostic.severity, HirDiagnosticSeverity::Error);
    assert_eq!(diagnostic.execution, DiagnosticExecution::RuntimeCheck);
    assert_eq!(diagnostic.primary.source, Some(ProgramSourceId(0)));
    let span = diagnostic.primary.span;
    assert!(source[span.start..span.end].contains("helper(A,B)"));
    assert_eq!(diagnostic.secondary[0].source, Some(ProgramSourceId(1)));
    assert!(diagnostic.suggestions.is_empty());
    let helper = mir
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name == runmat_hir::FunctionName("helper".into()))
        .map(|(function, _)| *function)
        .unwrap();
    assert!(before
        .function_for_local(helper)
        .unwrap()
        .callable
        .parameters
        .iter()
        .all(|fact| matches!(fact.kind, ValueKindFact::Unknown)));
    assert_eq!(
        before,
        crate::analysis::analyze_assembly(&mir),
        "diagnostic contexts must never alter generic facts or executable contracts"
    );
}

#[test]
fn compatible_scalar_and_distinct_valid_call_shapes_do_not_cross_contaminate() {
    for driver in [
        "C = helper(ones(2,3), ones(3,4));",
        "C = helper(2, ones(4,2));",
        "C = helper(ones(2,3), 2);",
        "C = helper(ones(2,3), ones(3,4)); D = helper(ones(5,2), ones(2,7));",
    ] {
        let mir = lower(&format!("{driver}{HELPER}"));
        let report = diagnose_call_contexts(&mir);
        assert!(!report.budget_exhausted);
        assert!(!report.precision_withheld);
        assert!(report.diagnostics.is_empty(), "{driver}");
    }
}

#[test]
fn unknown_recursive_expanded_and_environment_mutated_calls_stay_conservative() {
    for source in [
        format!("function C = driver(A,B)\nC = helper(A,B);\nend\n{HELPER}"),
        "C = helper(ones(2,3),ones(4,2));\nfunction C = helper(A,B)\nC = helper(A,B);\nC = A * B;\nend\n".into(),
        format!("args = {{ones(2,3),ones(4,2)}}; C = helper(args{{:}});{HELPER}"),
        format!("addpath('other'); C = helper(ones(2,3),ones(4,2));{HELPER}"),
    ] {
        let mir = lower(&source);
        let report = diagnose_call_contexts(&mir);
        assert!(!report.budget_exhausted);
        assert!(report.precision_withheld);
        assert!(report.diagnostics.is_empty(), "{source}");
    }
}

#[test]
fn invocation_arity_and_unknown_branch_guards_do_not_invent_caller_failures() {
    for source in [
        "C = helper(ones(2,3),ones(4,2));\nfunction [C,D] = helper(A,B)\nC = A;\nif nargout > 1\nD = A * B;\nend\nend\n",
        "C = helper(ones(2,3),ones(4,2));\nfunction C = helper(A,B)\nC = A;\nif nargin > 2\nC = A * B;\nend\nend\n",
        "C = helper(ones(2,3),ones(4,2),false);\nfunction C = helper(A,B,flag)\nC = A;\nif flag\nC = A * B;\nend\nend\n",
    ] {
        let report = diagnose_call_contexts(&lower(source));
        assert!(!report.budget_exhausted);
        assert!(report.precision_withheld);
        assert!(report.diagnostics.is_empty(), "{source}");
    }
}

#[test]
fn context_budget_withholds_precision_instead_of_changing_generic_analysis() {
    let mut source = String::new();
    for rows in 1..=MAX_CONTEXTS + 2 {
        source.push_str(&format!("C = helper(ones({rows},3),ones(4,2));\n"));
    }
    source.push_str(HELPER);
    let mir = lower(&source);
    let report = diagnose_call_contexts(&mir);
    assert!(report.budget_exhausted);
    assert!(report.precision_withheld);
    let diagnostics = report.diagnostics;
    assert_eq!(diagnostics.len(), MAX_CONTEXTS);
    assert!(diagnostics
        .iter()
        .all(|diagnostic| diagnostic.code == "RM-TYPE-MATMUL"));
}
