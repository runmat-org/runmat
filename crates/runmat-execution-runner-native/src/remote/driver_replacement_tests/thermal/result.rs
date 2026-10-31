use super::{Branch, Case};
use crate::{ExactStructuralStudyAnalysisResult, ExactStructuralStudyResult};
use runmat_analysis_core::StateUse;
use runmat_analysis_fea::execution::{
    CanonicalStateTransition, StructuralFrequencySolverPlan, StructuralLinearStateInput,
    StructuralLinearWorkload, StructuralLinearWorkloadResult,
};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactResult, ProgramExecutionResponse};

pub(in super::super) fn import_cold(
    case: Case,
    store: &impl CacheImport,
    response: ProgramExecutionResponse,
    expectation: &crate::StructuralStudyResultExpectation,
) -> ExactStructuralStudyResult {
    let limits = ObjectInventoryLimits::default();
    let mut foreign_source = expectation.clone();
    foreign_source.authored_study_digest = Digest::sha256(b"foreign authored thermal source");
    assert!(crate::import_structural_study_result(
        store,
        response.clone(),
        &foreign_source,
        limits
    )
    .is_err());
    let imported =
        crate::import_structural_study_result(store, response.clone(), expectation, limits)
            .unwrap();
    assert_authored_result(case, &imported);
    let thermal = imported.thermal.as_ref().unwrap();
    let thermal_root = runmat_analysis_execution::prepare_thermal_study_root(thermal).unwrap();
    // Unlike an empty-cache test, each exclusion removes one specifically required object
    // from an otherwise complete, verified cold closure. There is no producing-store fallback.
    for missing in [
        thermal_root.root.logical_digest,
        thermal.states[0].objects.root.digest,
    ] {
        assert!(store.read_verified(missing).unwrap().is_some());
        let incomplete = MissingObject {
            source: store,
            missing,
        };
        assert!(
            crate::import_structural_study_result(
                &incomplete,
                response.clone(),
                expectation,
                limits,
            )
            .is_err(),
            "{case:?}: accepted missing thermal object {missing:?}"
        );
    }
    imported
}

struct MissingObject<'a, T> {
    source: &'a T,
    missing: Digest,
}

impl<T: CacheImport> CacheImport for MissingObject<'_, T> {
    fn read_verified(&self, digest: Digest) -> ArtifactResult<Option<Vec<u8>>> {
        if digest == self.missing {
            Ok(None)
        } else {
            self.source.read_verified(digest)
        }
    }

    fn read_verified_bounded(
        &self,
        digest: Digest,
        maximum_bytes: u64,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        if digest == self.missing {
            Ok(None)
        } else {
            self.source.read_verified_bounded(digest, maximum_bytes)
        }
    }

    fn read_verified_bounded_with_admission(
        &self,
        digest: Digest,
        maximum_bytes: u64,
        admit: &mut dyn FnMut(u64) -> ArtifactResult<()>,
    ) -> ArtifactResult<Option<Vec<u8>>> {
        if digest == self.missing {
            Ok(None)
        } else {
            self.source
                .read_verified_bounded_with_admission(digest, maximum_bytes, admit)
        }
    }
}

pub(in super::super) fn assert_authored_result(case: Case, result: &ExactStructuralStudyResult) {
    let thermal = result
        .thermal
        .as_ref()
        .expect("cold typed root must retain thermal source");
    assert_eq!(result.compiled_study.as_ref(), Some(&thermal.compiled));
    assert_eq!(
        result.mesh.artifact.resolved_request.element_order,
        case.order
    );
    assert!(result
        .mesh
        .artifact
        .topology
        .volume_elements
        .iter()
        .all(|element| element.order == case.order));
    assert_eq!(
        thermal.preparation().mesh.canonical_digest,
        result.mesh.artifact.canonical_digest
    );
    assert_eq!(thermal.states.len(), 1);
    assert_eq!(
        thermal.states[0].state.state_id,
        runmat_analysis_core::StateId("heated".into())
    );
    let (workload, preload) = preload(&result.analysis);
    let source = preload
        .linearization
        .as_ref()
        .unwrap()
        .thermal_source()
        .unwrap();
    assert_eq!(
        source.transition().source_state,
        thermal.states[0].state.identity
    );
    source.validate_model(&thermal.compiled.model).unwrap();
    // Rebind from the independently imported thermal state, never its claimed transition.
    let mut bound = workload.clone();
    assert!(bound.state_input.is_none());
    assert!(preload.validate_thermal_workload(&bound).is_err());
    let transition = CanonicalStateTransition::bind_upstream(
        &bound.model,
        &bound.step_id,
        &thermal.states[0].state,
        StateUse::Prestress,
        Digest::from_bytes(*thermal.preparation().mesh.canonical_digest.bytes()),
    )
    .unwrap();
    assert_eq!(&transition, source.transition());
    bound.state_input = Some(StructuralLinearStateInput::ThermalPrestress(transition));
    preload.validate_thermal_workload(&bound).unwrap();
    assert_compression(result, preload);
    assert_branch(case, &result.analysis);
}

fn preload(
    analysis: &ExactStructuralStudyAnalysisResult,
) -> (&StructuralLinearWorkload, &StructuralLinearWorkloadResult) {
    match analysis {
        ExactStructuralStudyAnalysisResult::StructuralBuckling(value) => {
            (&value.preload_workload, &value.preload_result)
        }
        ExactStructuralStudyAnalysisResult::StructuralPrestressedModal(value) => {
            (&value.preload_workload, &value.preload_result)
        }
        ExactStructuralStudyAnalysisResult::StructuralPrestressedFrequency(value) => {
            (&value.preload_workload, &value.preload_result)
        }
        ExactStructuralStudyAnalysisResult::StructuralPrestressedModalFrequency(value) => {
            (&value.preload_workload, &value.preload_result)
        }
        other => panic!("expected a thermal prestress consumer, got {other:?}"),
    }
}

pub(in super::super) fn assert_branch(case: Case, analysis: &ExactStructuralStudyAnalysisResult) {
    match (case.branch, analysis) {
        (Branch::Buckling, ExactStructuralStudyAnalysisResult::StructuralBuckling(value)) => {
            value.buckling_result.validate().unwrap();
            value
                .buckling_numerical_receipt
                .validate_against(
                    value.buckling_numerical_receipt.source_result(),
                    &value.buckling_result,
                )
                .unwrap();
            assert!(value
                .buckling_result
                .modes()
                .iter()
                .any(|mode| mode.load_factor > 0.0));
            assert_eq!(
                value.prestress_state.source_result,
                value.preload_result_identity
            );
        }
        (Branch::Modal, ExactStructuralStudyAnalysisResult::StructuralPrestressedModal(value)) => {
            value.modal_result.validate().unwrap();
            value
                .modal_numerical_receipt
                .validate_against(
                    value.modal_numerical_receipt.source_result(),
                    &value.modal_result,
                )
                .unwrap();
            assert!(value
                .modal_result
                .basis()
                .modes()
                .iter()
                .all(|mode| mode.frequency_hz() > 0.0));
            assert_eq!(
                value.prestress_state.source_result,
                value.preload_result_identity
            );
        }
        (
            Branch::DirectHarmonic,
            ExactStructuralStudyAnalysisResult::StructuralPrestressedFrequency(value),
        ) => {
            assert!(matches!(
                value.frequency.resolved_workload.solver(),
                StructuralFrequencySolverPlan::Direct { .. }
            ));
            assert_eq!(
                value.prestress_state.source_result,
                value.preload_result_identity
            );
            assert_eq!(
                value
                    .frequency
                    .resolved_workload
                    .prestress_transition()
                    .unwrap()
                    .source_state,
                value.prestress_state.identity
            );
            assert_frequency(&value.frequency);
        }
        (
            Branch::ModalReducedHarmonic,
            ExactStructuralStudyAnalysisResult::StructuralPrestressedModalFrequency(value),
        ) => {
            value.modal_result.validate().unwrap();
            value
                .modal_numerical_receipt
                .validate_against(
                    value.modal_numerical_receipt.source_result(),
                    &value.modal_result,
                )
                .unwrap();
            assert_eq!(
                value.prestress_state.source_result,
                value.preload_result_identity
            );
            assert_eq!(
                value.basis_state.source_result,
                value.modal_numerical_receipt.source_result()
            );
            let StructuralFrequencySolverPlan::ModalReduced { basis_state } =
                value.frequency.resolved_workload.solver()
            else {
                panic!("expected retained reduced basis");
            };
            assert_eq!(*basis_state, value.basis_state.identity);
            assert_eq!(
                value
                    .frequency
                    .resolved_workload
                    .prestress_transition()
                    .unwrap()
                    .source_state,
                value.prestress_state.identity
            );
            assert_frequency(&value.frequency);
        }
        (expected, actual) => panic!("{case:?}: expected {expected:?}, got {actual:?}"),
    }
}

fn assert_frequency(frequency: &crate::ExactStructuralFrequencyAnalysisResult) {
    frequency.result.validate().unwrap();
    frequency
        .numerical_receipt
        .validate_for_result(&frequency.result)
        .unwrap();
    assert_eq!(frequency.result.samples().len(), 1);
    for sample in frequency.result.samples() {
        assert!(sample
            .displacement()
            .real()
            .iter()
            .any(|value| value.abs() > 1.0e-15));
        assert!(sample
            .displacement()
            .imaginary()
            .iter()
            .any(|value| value.abs() > 1.0e-18));
    }
}

fn assert_compression(
    result: &ExactStructuralStudyResult,
    preload: &StructuralLinearWorkloadResult,
) {
    let projection =
        runmat_analysis_fea::topology::AnalysisProjection::derive(&result.mesh.artifact).unwrap();
    let mut normals = Vec::new();
    let mut centers = Vec::new();
    let model = &result.thermal.as_ref().unwrap().compiled.model;
    for constraint in model
        .constraints
        .iter()
        .filter_map(|value| value.structural())
    {
        let target = constraint.kind.target().unwrap();
        let [identity] = target.resolved_entities() else {
            panic!("fixture support must bind one persistent face: {target:?}");
        };
        let faces = projection
            .boundary_faces
            .iter()
            .filter(|face| face.provenance.contains(identity))
            .collect::<Vec<_>>();
        assert!(!faces.is_empty());
        let area = faces
            .iter()
            .map(|face| face.area_m2.canonical_value())
            .sum::<f64>();
        let normal: [f64; 3] = std::array::from_fn(|axis| {
            faces
                .iter()
                .map(|face| face.oriented_area_vector_m2[axis])
                .sum::<f64>()
                / area
        });
        let center: [f64; 3] = std::array::from_fn(|axis| {
            faces
                .iter()
                .map(|face| face.centroid_m[axis] * face.area_m2.canonical_value())
                .sum::<f64>()
                / area
        });
        normals.push(normal);
        centers.push(center);
    }
    // Verify opposing x-normal supports from imported exact-CAD topology, not face names.
    assert_eq!(normals.len(), 2);
    assert!(normals
        .iter()
        .all(|normal| (normal[0].abs() - 1.0).abs() < 1.0e-12));
    assert!(normals[0][0] * normals[1][0] < 0.0);
    assert!((centers[0][0] - centers[1][0]).abs() > 0.5);
    let stress = preload
        .run
        .field(runmat_analysis_fea::FEA_FIELD_STRUCTURAL_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let mean_axial =
        stress.chunks_exact(6).map(|tensor| tensor[0]).sum::<f64>() / (stress.len() / 6) as f64;
    let material = &result.thermal.as_ref().unwrap().compiled.model.materials[0];
    let thermal_scale = material
        .require_mechanical()
        .unwrap()
        .youngs_modulus_pa
        .canonical_value()
        * material
            .require_thermal()
            .unwrap()
            .expansion_coefficient_per_k
            .canonical_value()
        * 10.0;
    assert!(
        mean_axial < -0.25 * thermal_scale,
        "thermal preload must produce meaningful compression: {mean_axial}"
    );
}
