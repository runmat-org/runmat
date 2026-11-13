//! Physical thermal-prestress chain regression, not independent full modal qualification.

use super::*;
use crate::execution::{
    SemanticStatePublicationRequest, SemanticStatePublicationSource, SequenceIdentity,
    StepIdentity, StructuralPrestressedModalTemplate,
};
use runmat_execution::Digest;

pub(super) fn thermal_preload(
    order: ElementOrder,
    delta: f64,
    consumer: ProcedureStep,
    configure: impl FnOnce(&mut AnalysisModel, &SolverMeshArtifact),
) -> (AnalysisModel, SolverMeshArtifact, CanonicalSemanticState) {
    let (mut model, mesh) = model_and_mesh(order, false);
    model
        .constraints
        .retain(|constraint| matches!(constraint, AnalysisConstraint::Thermal(_)));
    for constraint in &mut model.constraints {
        if let AnalysisConstraint::Thermal(ThermalConstraint {
            kind: ThermalConstraintKind::Prescribed { value, .. },
            ..
        }) = constraint
        {
            *value = PrescribedValue::Constant(Temperature::new(REFERENCE + delta));
        }
    }
    for node in &mesh.topology.nodes {
        // Same restraints in preload and modal consumers; no artificial support release.
        // All x DOFs fixed, but y/z remain free apart from three rigid-mode pins.
        let axes: &[usize] = if node.coordinates_m.canonical_value() == [0.0, 0.0, 0.0] {
            &[0, 1, 2]
        } else if node.coordinates_m.canonical_value() == [0.0, 1.0, 0.0] {
            &[0, 2]
        } else {
            &[0]
        };
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("axial_restraint_{}", node.node_id),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::point(format!("node:{}", node.node_id)),
                    translations: axes
                        .iter()
                        .map(|&axis| TranslationDofValue {
                            dof: [
                                StructuralTranslationDof::Ux,
                                StructuralTranslationDof::Uy,
                                StructuralTranslationDof::Uz,
                            ][axis],
                            value: PrescribedValue::Constant(Displacement::new(0.0)),
                        })
                        .collect(),
                    rotations: Vec::new(),
                }),
            }));
    }
    let declaration = ProcedureStep::structural_preload("declaration_template")
        .state_outputs
        .into_iter()
        .find(|output| output.kind == StateKind::GeometricStiffness)
        .unwrap();
    model.sequence.steps[1]
        .state_outputs
        .push(declaration.clone());
    let preload_id = model.sequence.steps[1].step_id.clone();
    let thermal_transition = model.sequence.transitions[0].clone();
    let mut steps = model.sequence.steps.clone();
    steps.push(consumer.clone());
    model.sequence = AnalysisSequence::from_steps("thermal_static_modal_qualification", steps);
    for dependency in &mut model.sequence.dependencies {
        dependency.reason = SequenceDependencyReason::StateTransfer;
    }
    model.sequence.transitions = vec![
        thermal_transition,
        StateTransition {
            transition_id: TransitionId("thermal_preload_to_modal".into()),
            producer_step: preload_id,
            consumer_step: consumer.step_id.clone(),
            state_id: declaration.state_id.clone(),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        },
    ];
    configure(&mut model, &mesh);
    let thermal = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::Steady,
            solver_mesh: Some(mesh.clone()),
            ..ThermalSolveOptions::default()
        },
    )
    .unwrap();
    let source = CanonicalSemanticState::from_thermal_result(&model, &mesh, &thermal).unwrap();
    let retained_source = source.clone();
    let preload =
        run_linear_static_from_thermal_state(&model, ComputeBackend::Cpu, options(&mesh), &source)
            .unwrap();
    assert_eq!(source, retained_source);
    let mechanical = model.materials[0].require_mechanical().unwrap();
    let axial_stress = -mechanical.youngs_modulus_pa.canonical_value() * ALPHA * delta;
    let stress_scale = mechanical.youngs_modulus_pa.canonical_value() * ALPHA * DELTA;
    let stress = field(&preload.run, crate::FEA_FIELD_STRUCTURAL_STRESS);
    assert_eq!(stress.len(), 6);
    assert!((stress[0] - axial_stress).abs() < stress_scale * 1.0e-7);
    assert!(stress[1..]
        .iter()
        .all(|value| value.abs() < stress_scale * 1.0e-7));
    let displacement = field(&preload.run, crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert_eq!(displacement.len(), mesh.topology.nodes.len() * 3);
    let dilation = ALPHA * delta * (1.0 + mechanical.poisson_ratio.canonical_value());
    for (node, u) in mesh.topology.nodes.iter().zip(displacement.chunks_exact(3)) {
        assert!(u[0].abs() < 1.0e-12);
        for (axis, value) in u.iter().enumerate().skip(1) {
            assert!(
                (value - dilation * node.coordinates_m.canonical_value()[axis]).abs() < 1.0e-11
            );
        }
    }
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[1]).unwrap();
    let publication = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralLinearization,
        producer,
        declaration,
        topology,
        preload.solver_result_identity,
    )
    .unwrap();
    let geometric =
        CanonicalSemanticState::from_linearization_publication(&publication, preload.linearization)
            .unwrap();
    let portable =
        CanonicalSemanticState::from_canonical_bytes(&geometric.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(portable, geometric);
    assert_eq!(source, retained_source);
    (model, mesh, portable)
}

fn thermal_prestressed_frequency(order: ElementOrder, delta: f64) -> f64 {
    let mut modal = ProcedureStep::structural_prestressed_modal(
        "thermal_prestressed_modes",
        "qualification_expansion",
        "geometric_stiffness",
    );
    modal.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
    let (model, mesh, portable) = thermal_preload(order, delta, modal.clone(), |_, _| {});
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let retained_geometric = portable.clone();
    let transition = CanonicalStateTransition::bind_upstream(
        &model,
        &modal.step_id,
        &portable,
        StateUse::Prestress,
        topology,
    )
    .unwrap();
    let workload = StructuralPrestressedModalTemplate {
        schema_version: 1,
        model,
        step_id: modal.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: crate::ModalTruncationPolicy::requested(1),
        convergence: crate::ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
    }
    .bind(transition)
    .unwrap();
    let result =
        crate::execute_structural_prestressed_modal_workload(&workload, &mesh, &portable).unwrap();
    assert_eq!(portable, retained_geometric);
    let frequency = result.basis().modes()[0].frequency_hz();
    assert!(frequency.is_finite() && frequency > 0.0);
    frequency
}

#[test]
fn actual_thermal_preload_softens_heating_and_stiffens_cooling_modes() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let unloaded = thermal_prestressed_frequency(order, 0.0);
        let heated = thermal_prestressed_frequency(order, DELTA);
        let cooled = thermal_prestressed_frequency(order, -DELTA);
        assert!(
            heated < unloaded,
            "{order:?}: heating {heated}, reference {unloaded}"
        );
        assert!(
            cooled > unloaded,
            "{order:?}: cooling {cooled}, reference {unloaded}"
        );
    }
}
