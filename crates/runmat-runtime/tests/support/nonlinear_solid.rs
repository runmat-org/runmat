//! A canonical Tet4 solid for public nonlinear result-contract checks.
use runmat_analysis_core::*;
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder, StableDigest};
use runmat_types::physical::Quantity;

pub fn fixture() -> (
    tempfile::TempDir,
    AnalysisModel,
    runmat_runtime::analysis::AnalysisNonlinearRunOptions,
) {
    let root = tempfile::tempdir().expect("nonlinear contract mesh root");
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::NonlinearAssembly,
    );
    model.structural = None;
    model.connectors.clear();
    model.interfaces.clear();
    model.constraints.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.materials.truncate(1);
    let material = &mut model.materials[0];
    material.elasticity = Some(MaterialElasticityModel::IsotropicLinear(
        MaterialMechanicalModel {
            youngs_modulus_pa: Quantity::new(1200.0),
            poisson_ratio: Quantity::new(0.25),
        },
    ));
    material.plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
        yield_stress_pa: Quantity::new(5.0),
        isotropic_hardening_modulus_pa: Quantity::new(30.0),
    });
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    model.material_assignments = vec![MaterialAssignment {
        target: PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_region(
            "region",
        )),
        expected_material_id: material.material_id.clone(),
        assigned_material_id: material.material_id.clone(),
        confidence: EvidenceConfidence::Verified,
    }];
    for (axis, dof) in [
        StructuralTranslationDof::Ux,
        StructuralTranslationDof::Uy,
        StructuralTranslationDof::Uz,
    ]
    .into_iter()
    .enumerate()
    {
        let target =
            runmat_meshing_core::fixtures::fixture_face(&format!("contract_anchor_{axis}"));
        let topology = &mut *mesh.topology;
        let face = topology
            .boundary_faces
            .iter_mut()
            .find(|face| {
                face.node_ids.iter().all(|id| {
                    topology
                        .nodes
                        .iter()
                        .find(|node| node.node_id == *id)
                        .unwrap()
                        .coordinates_m
                        .canonical_value()[axis]
                        == 0.0
                })
            })
            .expect("coordinate-plane anchor");
        face.provenance.push(
            runmat_meshing_core::fixtures::fixture_entity_for_selection(&target)
                .expect("canonical anchor face identity"),
        );
        model
            .constraints
            .push(AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("contract_anchor_{axis}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::boundary(target),
                    translations: vec![TranslationDofValue {
                        dof,
                        value: PrescribedValue::Constant(Displacement::new(0.0)),
                    }],
                    rotations: vec![],
                }),
            }));
    }
    let target = runmat_meshing_core::fixtures::fixture_face("contract_loaded");
    let topology = &mut *mesh.topology;
    topology
        .boundary_faces
        .iter_mut()
        .find(|face| {
            face.node_ids.iter().all(|id| {
                (topology
                    .nodes
                    .iter()
                    .find(|node| node.node_id == *id)
                    .unwrap()
                    .coordinates_m
                    .canonical_value()
                    .iter()
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1e-14
            })
        })
        .expect("inclined loaded face")
        .provenance
        .push(
            runmat_meshing_core::fixtures::fixture_entity_for_selection(&target)
                .expect("canonical loaded face identity"),
        );
    for face in &mut topology.boundary_faces {
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest()
        .expect("seal contract solid mesh");
    let signal = SignalId("contract_traction".into());
    model.signals.push(PhysicalSignal::Traction(Signal {
        signal_id: signal.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3([
            Traction::new(5.6 / 3.0_f64.sqrt()),
            Traction::new(0.0),
            Traction::new(0.0),
        ])),
        precision: SignalPrecision::Real64,
        provenance: SignalProvenance::Authored,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
    }));
    model
        .excitations
        .push(ExcitationAction::Traction(AnalysisExcitation {
            excitation_id: ExcitationId("contract_traction".into()),
            target: ExcitationTarget::Region(RegionTarget::boundary(target)),
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::IntensityPerArea,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: signal,
            active_step_ids: vec![],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: false,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        }));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("contract_traction".into()),
        action_ids: vec![ExcitationId("contract_traction".into())],
        activation: ExcitationCaseActivation::Always,
    });
    let path = root.path().join("nonlinear_contract.cbor");
    std::fs::write(
        &path,
        mesh.canonical_encode().expect("encode contract solid mesh"),
    )
    .unwrap();
    (
        root,
        model,
        runmat_runtime::analysis::AnalysisNonlinearRunOptions {
            solver_mesh_artifact_path: Some(path.to_string_lossy().into_owned()),
            increment_count: 4,
            tolerance: 1e-9,
            linear_tolerance: 1e-12,
            ..Default::default()
        },
    )
}
