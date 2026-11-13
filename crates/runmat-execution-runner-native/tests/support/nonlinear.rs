//! Shared original homogeneous J2 witness fixture; not a production operator or reference generator.
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::NonlinearSolidWorkload,
    fixtures::{fixture_model, FixtureId},
    solve::nonlinear::NonlinearSolveOptions,
    study::CompiledAnalysisStudy,
};
use runmat_meshing_core::{
    fixtures::{quadratic_slab_mesh, slab_mesh},
    SolverMeshArtifact,
};
pub(crate) fn fixture(quadratic: bool) -> (NonlinearSolidWorkload, SolverMeshArtifact) {
    let mesh = if quadratic {
        quadratic_slab_mesh(1)
    } else {
        slab_mesh(1)
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.materials[0].elasticity = Some(MaterialElasticityModel::IsotropicLinear(
        MaterialMechanicalModel {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(1000.0),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(0.25),
        },
    ));
    model.materials[0].plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
        yield_stress_pa: runmat_types::physical::YieldStress::new(5.0),
        isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(30.0),
    });
    model.sequence = AnalysisSequence::from_steps(
        "native-j2",
        vec![ProcedureStep::structural_nonlinear("plastic")],
    );
    model.signals.push(PhysicalSignal::Traction(Signal {
        signal_id: SignalId("pull".into()),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3([
            Traction::new(5.6),
            Traction::new(0.0),
            Traction::new(0.0),
        ])),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }));
    model
        .excitations
        .push(ExcitationAction::Traction(AnalysisExcitation {
            excitation_id: ExcitationId("pull".into()),
            target: ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("slab_right"),
            )),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerArea,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId("pull".into()),
            active_step_ids: vec!["plastic".into()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        }));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("pull".into()),
        action_ids: vec![ExcitationId("pull".into())],
        activation: ExcitationCaseActivation::Always,
    });
    for node in &mesh.topology.nodes {
        let translations = [
            StructuralTranslationDof::Ux,
            StructuralTranslationDof::Uy,
            StructuralTranslationDof::Uz,
        ]
        .into_iter()
        .enumerate()
        .filter(|(axis, _)| node.coordinates_m.canonical_value()[*axis] == 0.0)
        .map(|(_, dof)| TranslationDofValue {
            dof,
            value: PrescribedValue::Constant(Displacement::new(0.0)),
        })
        .collect::<Vec<_>>();
        if !translations.is_empty() {
            model
                .constraints
                .push(AnalysisConstraint::Structural(StructuralConstraint {
                    constraint_id: format!("pin_{}", node.node_id),
                    frame: CoordinateFrame::Global,
                    kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                        target: RegionTarget::point(format!("node:{}", node.node_id)),
                        translations,
                        rotations: Vec::new(),
                    }),
                }));
        }
    }
    (
        NonlinearSolidWorkload {
            model,
            controls: NonlinearSolveOptions {
                increment_count: 10,
                max_accepted_increments: 12,
                max_newton_iters: 32,
                tolerance: 1e-9,
                linear_tolerance: 1e-12,
                ..NonlinearSolveOptions::default()
            },
        },
        mesh,
    )
}

pub(crate) fn compiled(
    workload: &NonlinearSolidWorkload,
    mesh: &SolverMeshArtifact,
) -> CompiledAnalysisStudy {
    // Use contract decoding for optional metadata, so this witness does not depend
    // on unrelated uncommitted additions to the compiled-study struct.
    serde_json::from_value(serde_json::json!({
        "module_path": "tests/original-j2.fea",
        "study_id": "plastic",
        "study_inputs": [], "outputs": [],
        "model": workload.model,
        "properties": runmat_analysis_fea::properties::compile_property_map(&workload.model, Some(mesh)).unwrap(),
        "solver_mesh_digest": mesh.canonical_digest,
    })).unwrap()
}
