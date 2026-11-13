//! Algebraic policy checks; final physical mesh validation is independent.
use super::*;
use crate::cdt::test_fixtures::coupled_quality::{fixture, metric};
use runmat_meshing_core::NeverCancelled;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Canonical coupled admission accepts a strictly lower combined size/shape defect and
//   rejects a reverse trade under either primary source objective; the standalone shape treatment
//   still preserves refinement. The reference comparator independently merges cell records. This is
//   an algebraic policy witness, not a terminal physical-mesh or original-study qualification.
#[test]
fn coupled_canonical_admission_uses_joint_order_while_standalone_preserves_refinement() {
    let (topology, provenance) = fixture(false, 0);
    let metric = metric(&provenance, false);
    let options = crate::cdt::DelaunayVolumeQualityOptions::default();
    let mut before = crate::cdt::evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    for cell in &mut before.tetrahedra {
        cell.refinement_violation_ratio = 0.5;
        cell.optimization_violation_ratio = 0.5;
    }
    before.tetrahedra[0].optimization_violation_ratio = 9.;
    let mut transfer = before.clone();
    transfer.tetrahedra[0].optimization_violation_ratio = 0.5;
    transfer.tetrahedra[0].refinement_violation_ratio = 8.;
    let mut undo = transfer.clone();
    undo.tetrahedra[0].optimization_violation_ratio = 8.5;
    undo.tetrahedra[0].refinement_violation_ratio = 0.5;
    let base_input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &metric,
        provenance: &provenance,
        quality: &before,
        quality_options: options,
        fixed_node_identities: &[],
    };
    let transfer_input = DelaunayVolumeRefinementInput {
        quality: &transfer,
        ..base_input
    };
    for objective in [QualityObjective::Shape, QualityObjective::Refinement] {
        let coupled = CandidateRefinementPolicy::AllowForCoupledConvergence;
        let baseline = candidate_spectrum(&before, coupled, objective);
        let next = candidate_spectrum(&transfer, coupled, objective);
        assert_eq!(
            compare_spectra(&next, &baseline),
            crate::cdt::volume_quality::joint::compare(&before, &transfer)
        );
        assert!(candidate_quality_is_admissible(
            &transfer,
            base_input,
            &baseline,
            coupled,
            objective,
            &mut BTreeMap::new()
        ));
        assert!(!candidate_quality_is_admissible(
            &undo,
            transfer_input,
            &next,
            coupled,
            objective,
            &mut BTreeMap::new()
        ));
    }
    assert!(!candidate_quality_is_admissible(
        &transfer,
        base_input,
        &quality_spectrum(&before, QualityObjective::Shape),
        CandidateRefinementPolicy::Preserve,
        QualityObjective::Shape,
        &mut BTreeMap::new()
    ));
}
