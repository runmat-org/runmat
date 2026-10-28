use super::*;
use runmat_analysis_core::{
    validate_model, validate_model_sequence, SequenceDependency, SequenceDependencyReason,
};

#[test]
fn original_model_checks_and_owned_planning_topology() {
    // Regenerated deliberately after the formulation-maturity field left the authored step
    // encoding (catalog-port b6501a93d2), and again when the authored memory ceiling default
    // became unbounded, and again when an unbounded ceiling became an explicit null (wire
    // integers bounded to 2^53 - 1); the owned/checked graphs still agree byte for byte.
    let expected = [
        "sha256:3790cd03b9feac2136dd4ca72ecfe6b81af164ec6f67a466fccd76b5086b959d",
        "sha256:40c486ec16bf5437e9dafccd9a2a137f64c67780547d04cdaab1e3cdcdd03576",
        "sha256:9c4bf1b9ab3f0aef866c85129944fb350fc1020e68112e3b54828a4487c2eafe",
    ];
    let mut actual = Vec::new();
    for count in [1, 4, 32] {
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        let original = model.sequence.steps[0].clone();
        for index in 1..count {
            let mut step = original.clone();
            step.step_id = ProcedureStepId(format!("step_{index:03}"));
            model.sequence.steps.push(step);
        }
        // A fan-out/fan-in DAG exercises deterministic parallel levels and direct dependencies.
        if count > 1 {
            for index in 1..count - 1 {
                model.sequence.dependencies.push(SequenceDependency {
                    predecessor: original.step_id.clone(),
                    successor: model.sequence.steps[index].step_id.clone(),
                    reason: SequenceDependencyReason::AuthoredOrdering,
                });
                model.sequence.dependencies.push(SequenceDependency {
                    predecessor: model.sequence.steps[index].step_id.clone(),
                    successor: model.sequence.steps[count - 1].step_id.clone(),
                    reason: SequenceDependencyReason::AuthoredOrdering,
                });
            }
        }
        let (_, checked_live, checked_peak) =
            crate::topology::measure_test_allocations(|| validate_model(&model).unwrap());
        let (owned, owned_live, owned_peak) =
            crate::topology::measure_test_allocations(|| validate_model_sequence(&model).unwrap());
        assert_eq!(checked_live, 0);
        assert!(owned_live > 0);
        assert!(checked_peak < owned_peak);
        assert_eq!(owned.sequence(), &model.sequence);
        let identity = Digest::sha256(
            encode_contract(
                b"validation-topology-reference\0",
                "sequence",
                &owned,
                LIMITS,
            )
            .unwrap(),
        );
        actual.push(identity.to_string());
        println!("MODEL_VALIDATION count={count} checked_peak={checked_peak} owned_peak={owned_peak} owned_live={owned_live}");
    }
    assert_eq!(actual, expected);
}

#[test]
fn model_check_and_owned_planning_reject_original_semantic_errors() {
    use runmat_analysis_core::{AnalysisValidationError, RestartPolicy};
    for case in 0..6 {
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        match case {
            // Material failure precedes graph failure at both public boundaries.
            0 => {
                model.materials.clear();
                model.sequence.steps.push(model.sequence.steps[0].clone());
            }
            1 => {
                model.constraints.clear();
                model.boundaries.clear();
            }
            2 => model.sequence.steps.push(model.sequence.steps[0].clone()),
            3 | 4 => model.sequence.dependencies.push(SequenceDependency {
                predecessor: model.sequence.steps[0].step_id.clone(),
                successor: if case == 3 {
                    ProcedureStepId("absent".into())
                } else {
                    model.sequence.steps[0].step_id.clone()
                },
                reason: SequenceDependencyReason::AuthoredOrdering,
            }),
            5 => {
                model.sequence.steps[0].restart = RestartPolicy::ResumeCheckpoint {
                    checkpoint_id: "broken".into(),
                    required_state_digest: String::new(),
                }
            }
            _ => unreachable!(),
        }
        let checked = validate_model(&model).unwrap_err();
        let owned = validate_model_sequence(&model).unwrap_err();
        assert_eq!(checked, owned);
        match case {
            0 => assert!(matches!(checked, AnalysisValidationError::MissingMaterials)),
            1 => assert!(matches!(
                checked,
                AnalysisValidationError::MissingConstraintsOrBoundaries
            )),
            2 | 5 => assert!(matches!(
                checked,
                AnalysisValidationError::InvalidProcedureStep { .. }
            )),
            3 | 4 => assert!(matches!(
                checked,
                AnalysisValidationError::InvalidSequence { .. }
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn model_validation_chain_workspace_grows_with_original_graph_inventory() {
    let mut peaks = Vec::new();
    for count in [32, 256] {
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        let template = model.sequence.steps[0].clone();
        for index in 1..count {
            let mut step = template.clone();
            step.step_id = ProcedureStepId(format!("chain-{index}"));
            model.sequence.steps.push(step);
            model.sequence.dependencies.push(SequenceDependency {
                predecessor: model.sequence.steps[index - 1].step_id.clone(),
                successor: model.sequence.steps[index].step_id.clone(),
                reason: SequenceDependencyReason::AuthoredOrdering,
            });
        }
        let (_, live, peak) =
            crate::topology::measure_test_allocations(|| validate_model(&model).unwrap());
        assert_eq!(live, 0);
        peaks.push(peak);
        println!("MODEL_CHAIN count={count} peak={peak}");
    }
    // Eight times the original vertices/edges; allow hash-table capacity thresholds.
    // This checks growth of the observed component, not admission or RSS.
    assert!(peaks[1] <= 12 * peaks[0]);
}

#[test]
fn dependency_graph_component_budget_is_exact_and_precedes_validation() {
    use runmat_analysis_core::{
        dependency_graph_resident_bytes_upper_bound, validate_model_with_dependency_graph_budget,
        AnalysisValidationError,
    };
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let required = dependency_graph_resident_bytes_upper_bound(&model.sequence).unwrap();
    validate_model_with_dependency_graph_budget(&model, required).unwrap();
    model.materials.clear();
    assert!(
        matches!(validate_model_with_dependency_graph_budget(&model, required-1), Err(AnalysisValidationError::DependencyGraphResourceLimit { requested_bytes, maximum_bytes }) if requested_bytes == required && maximum_bytes == required-1)
    );
    assert!(matches!(
        validate_model_with_dependency_graph_budget(&model, required),
        Err(AnalysisValidationError::MissingMaterials)
    ));
}
