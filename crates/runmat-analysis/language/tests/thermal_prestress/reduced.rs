use super::{check, SOURCE};
use runmat_analysis_core::{AnalysisProcedure, SolutionSpace, StateKind, StateSource, StateUse};

fn source(bindings: &str, after: &str) -> String {
    format!(
        "{}\nanalysis response after analysis.{after} {{\nstructural prestressed_harmonic {{\n{bindings}\nfrequencies = [60 Hz]\n}}\n}}\n}}",
        SOURCE.trim_end().strip_suffix('}').unwrap()
    )
}

const BINDINGS: &str = "prestress = analysis.preload.state.geometric_stiffness\nbasis = analysis.modes.state.modal_basis";

#[test]
fn authored_reduced_prestressed_response_retains_both_state_dependencies() {
    for after in ["modes", "preload"] {
        let checked = check(&source(BINDINGS, after));
        assert!(checked.is_valid(), "{:#?}", checked.diagnostics);
        let resolved = checked
            .resolve_study("project/main.fea", "response")
            .unwrap();
        let realized = resolved
            .study
            .realize_with_models(&resolved.models, &resolved.geometry, &[])
            .unwrap();
        let step = &realized.sequence.steps[3];
        assert_eq!(
            step.procedure,
            AnalysisProcedure::PrestressedHarmonicResponse
        );
        assert_eq!(step.solution_space, SolutionSpace::ModalReduced);
        assert_eq!(step.state_inputs.len(), 2);
        for (kind, use_as, producer) in [
            (
                StateKind::GeometricStiffness,
                StateUse::Prestress,
                "preload",
            ),
            (StateKind::ModalBasis, StateUse::ModalBasis, "modes"),
        ] {
            assert!(step.state_inputs.iter().any(|input| input.required_kind == kind
                && input.use_as == use_as
                && matches!(&input.source, StateSource::Upstream { producer_step, .. } if producer_step.as_str() == producer)));
            assert!(realized
                .sequence
                .dependencies
                .iter()
                .any(|edge| edge.predecessor.as_str() == producer
                    && edge.successor.as_str() == "response"));
            assert!(realized
                .sequence
                .transitions
                .iter()
                .any(|edge| edge.producer_step.as_str() == producer
                    && edge.consumer_step.as_str() == "response"));
        }
    }
}

#[test]
fn reduced_prestress_rejects_invalid_or_ambiguous_state_bindings() {
    for bindings in [
        BINDINGS.replace(
            "analysis.modes.state.modal_basis",
            "analysis.heat.state.heated",
        ),
        BINDINGS.replace(
            "analysis.modes.state.modal_basis",
            "analysis.missing.state.modal_basis",
        ),
        BINDINGS.replace(
            "analysis.modes.state.modal_basis",
            "analysis.modes.state.absent",
        ),
        BINDINGS.replace("analysis.modes.state.modal_basis", "1"),
        format!("{BINDINGS}\nbasis = analysis.modes.state.modal_basis"),
        format!("{BINDINGS}\nunknown = 1"),
    ] {
        assert!(
            !check(&source(&bindings, "modes")).is_valid(),
            "accepted {bindings}"
        );
    }
    assert!(!check(&source(BINDINGS, "heat")).is_valid());
}
