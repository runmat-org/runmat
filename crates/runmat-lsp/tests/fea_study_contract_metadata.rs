use lsp_types::{Position, SignatureHelp};
use runmat_builtins::BuiltinCompletionPolicy;
use runmat_lsp::core::analysis::{
    analyze_document_with_compat, completion_at, signature_help_at, CompatMode,
};

const REPRESENTATIVE_STUDY_CONTRACTS: &[&str] = &[
    "fea.IndependentDomain.timeSeconds",
    "fea.StepObjectRef.boundary",
    "fea.ActivationLifecycle.modify",
    "fea.StepObjectActivation.new",
    "fea.StateInput.upstream",
    "fea.StateOutput.new",
    "fea.SequenceDependency.new",
    "fea.StateTransition.new",
    "fea.SequenceBranch.new",
    "fea.RestartPolicy.resumeCheckpoint",
    "fea.Observation.field",
    "fea.Observation.monitor",
    "fea.ReductionPolicy.none",
    "fea.AccuracyCriterion.new",
    "fea.AccuracyPlan.new",
    "fea.ProcedureStep.configure",
    "fea.AnalysisSequence.new",
    "fea.ComputeBackend.cpu",
    "fea.SolverOptions.defaults",
    "fea.ExecutionStep.new",
    "fea.ExecutionPlan.new",
    "fea.ExecutionPlan.defaults",
];

fn signature_help(source: &str, lookup: &str) -> SignatureHelp {
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    assert!(analysis.syntax_error.is_none(), "{source}");
    assert!(
        analysis.lowering_error.is_none(),
        "{source}: {:?}",
        analysis.lowering_error
    );
    assert!(
        analysis.compile_error.is_none(),
        "{source}: {:?}",
        analysis.compile_error
    );
    let column = source.find(lookup).expect("builtin call") as u32;
    signature_help_at(source, &analysis, &Position::new(0, column))
        .unwrap_or_else(|| panic!("descriptor-backed signature help for {lookup}"))
}

#[test]
fn typed_study_contracts_are_public_and_documented() {
    let source = "x = 1;";
    let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
    let completions = completion_at(source, &analysis, &Position::new(0, 0));

    for name in REPRESENTATIVE_STUDY_CONTRACTS {
        // Typed study contracts are catalog-backed: the catalog entry owns the
        // public descriptor that completion and signature help read.
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name)
            .unwrap_or_else(|| panic!("catalog entry for {name}"));
        let descriptor = entry.descriptor;
        assert_eq!(
            descriptor.completion_policy,
            BuiltinCompletionPolicy::Public,
            "{name} completion policy"
        );
        assert!(!descriptor.signatures.is_empty(), "{name} signatures");
        assert!(
            completions
                .iter()
                .any(|item| item.label.eq_ignore_ascii_case(name)),
            "completion for {name}"
        );
    }
}

#[test]
fn sequence_and_observation_signatures_are_available_during_authoring() {
    for (lookup, source) in [
        (
            "ProcedureStep.configure",
            "step = fea.ProcedureStep.configure(0, 0, 0, 0, 0, 0, 0, 0);",
        ),
        (
            "Observation.field",
            "request = fea.Observation.field(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);",
        ),
        (
            "AnalysisSequence.new",
            "sequence = fea.AnalysisSequence.new(0, 0, 0, 0, 0);",
        ),
        (
            "ExecutionStep.new",
            "executionStep = fea.ExecutionStep.new(0, 0, 0, 0, 0);",
        ),
        (
            "ExecutionPlan.new",
            "execution = fea.ExecutionPlan.new(0, 0);",
        ),
    ] {
        let help = signature_help(source, lookup);
        assert!(
            help.signatures
                .iter()
                .any(|signature| signature.label.contains(&format!("fea.{lookup}"))),
            "signature for {lookup}"
        );
    }
}
