//! Original accepted nonlinear evidence remains bound through cold receipt/state publication.
use super::*;
use crate::execution::{
    CanonicalSemanticState, SemanticStatePublicationRequest, SemanticStatePublicationSource,
    ThermalNumericalReceiptArtifact, ThermalSolverResultManifest,
};

pub(super) fn verify(
    input: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    result: &FeaThermalRunResult,
) {
    let expected = match input.procedure {
        ThermalProcedureControl::Steady => TemporalImplementation::ThermalSteadyDampedNewton,
        ThermalProcedureControl::BackwardEuler { .. } => {
            TemporalImplementation::ThermalBackwardEulerDampedNewton
        }
    };
    assert_eq!(result.run.identity.realization().temporals, vec![expected]);
    let manifest = ThermalSolverResultManifest::from_result(result).unwrap();
    let manifest =
        ThermalSolverResultManifest::from_canonical_bytes(&manifest.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(manifest.nonlinear_evidence, result.nonlinear_evidence);
    let receipt = ThermalNumericalReceiptArtifact::new(input, result).unwrap();
    assert_eq!(
        receipt,
        ThermalNumericalReceiptArtifact::from_manifest(input, &manifest).unwrap()
    );
    let receipt =
        ThermalNumericalReceiptArtifact::from_canonical_bytes(&receipt.canonical_bytes().unwrap())
            .unwrap();
    receipt.validate_against(input, &manifest).unwrap();
    assert_eq!(
        receipt.source_result(),
        result.source_result_identity().unwrap()
    );

    let declaration = input.model.sequence.steps[0].state_outputs[0].clone();
    let publication = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::ThermalPrimaryField,
        result.producer,
        declaration,
        result.topology_digest,
        result.source_result_identity().unwrap(),
    )
    .unwrap();
    let publication = SemanticStatePublicationRequest::from_canonical_bytes(
        &publication.canonical_bytes().unwrap(),
    )
    .unwrap();
    let state = CanonicalSemanticState::from_thermal_publication(&publication, result).unwrap();
    assert_eq!(
        state,
        CanonicalSemanticState::from_thermal_result(&input.model, mesh, result).unwrap()
    );
    assert_eq!(state.producer, result.producer);
    assert_eq!(state.source_result, manifest.identity());
    assert_eq!(
        CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap()).unwrap(),
        state
    );

    for invalid in 0..3 {
        let mut changed = result.clone();
        match invalid {
            0 => changed.nonlinear_evidence.clear(),
            1 => changed.nonlinear_evidence[0].relative_increment = 2.0 * TARGET,
            _ => changed.nonlinear_evidence[0].relative_equilibrium_residual = 2.0 * TARGET,
        }
        assert!(changed.validate_complete().is_err());
        assert!(ThermalNumericalReceiptArtifact::new(input, &changed).is_err());
        assert!(CanonicalSemanticState::from_thermal_publication(&publication, &changed).is_err());
        let mut changed = manifest.clone();
        match invalid {
            0 => changed.nonlinear_evidence.clear(),
            1 => changed.nonlinear_evidence[0].relative_increment = 2.0 * TARGET,
            _ => changed.nonlinear_evidence[0].relative_equilibrium_residual = 2.0 * TARGET,
        }
        assert!(ThermalNumericalReceiptArtifact::from_manifest(input, &changed).is_err());
    }
    let mut changed = result.clone();
    changed.nonlinear_evidence[0].backtracking_count += 1;
    changed.validate_complete().unwrap();
    assert_ne!(
        changed.source_result_identity().unwrap(),
        publication.source_result
    );
    assert!(CanonicalSemanticState::from_thermal_publication(&publication, &changed).is_err());
    let changed = ThermalSolverResultManifest::from_result(&changed).unwrap();
    assert!(receipt.validate_against(input, &changed).is_err());
}
