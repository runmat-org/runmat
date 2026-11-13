use super::*;

fn signal<Q>(id: &str, values: [Q; 3]) -> Signal<Q> {
    Signal {
        signal_id: SignalId(id.into()),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3(values)),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }
}

pub(crate) fn prepare(
    cells: usize,
    quadratic: bool,
) -> (AnalysisModel, SolverMeshArtifact, AssemblySummary) {
    prepare_mesh(if quadratic {
        quadratic_slab_mesh(cells)
    } else {
        slab_mesh(cells)
    })
}

/// The same uniaxial-strain fixture on a caller-supplied unit-cube slab mesh.
pub(crate) fn prepare_mesh(
    mesh: SolverMeshArtifact,
) -> (AnalysisModel, SolverMeshArtifact, AssemblySummary) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.signals.clear();
    model.excitation_cases.clear();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.materials[0].elasticity = Some(MaterialElasticityModel::IsotropicLinear(
        MaterialMechanicalModel {
            youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(E),
            poisson_ratio: runmat_types::physical::PoissonRatio::new(NU),
        },
    ));
    model.materials[0].plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
        yield_stress_pa: runmat_types::physical::YieldStress::new(YIELD),
        isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(HARDENING),
    });
    model.sequence = AnalysisSequence::from_steps(
        "manufactured_nonlinear_cube",
        vec![ProcedureStep::structural_nonlinear("plastic")],
    );
    let body = -tangent() * B;
    let traction = stress(1.0)[0];
    model.signals = vec![
        PhysicalSignal::BodyForceDensity(signal(
            "body",
            [
                BodyForceDensity::new(body),
                BodyForceDensity::new(0.0),
                BodyForceDensity::new(0.0),
            ],
        )),
        PhysicalSignal::Traction(signal(
            "traction",
            [
                Traction::new(traction),
                Traction::new(0.0),
                Traction::new(0.0),
            ],
        )),
    ];
    for (id, target, distribution) in [
        (
            "body",
            ExcitationTarget::WholeModel,
            ExcitationDistribution::IntensityPerVolume,
        ),
        (
            "traction",
            ExcitationTarget::Region(RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("slab_right"),
            )),
            ExcitationDistribution::IntensityPerArea,
        ),
    ] {
        let action = AnalysisExcitation {
            excitation_id: ExcitationId(id.into()),
            target,
            frame: CoordinateFrame::Global,
            direction: None,
            distribution,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId(id.into()),
            active_step_ids: vec!["plastic".into()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        };
        model.excitations.push(if id == "body" {
            ExcitationAction::BodyForce(action)
        } else {
            ExcitationAction::Traction(action)
        });
    }
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("dead_loads".into()),
        action_ids: vec![ExcitationId("body".into()), ExcitationId("traction".into())],
        activation: ExcitationCaseActivation::Always,
    }];
    for node in &mesh.topology.nodes {
        let translations = [
            StructuralTranslationDof::Ux,
            StructuralTranslationDof::Uy,
            StructuralTranslationDof::Uz,
        ]
        .into_iter()
        .filter(|d| {
            node.coordinates_m.canonical_value()[0] == 0.0 || *d != StructuralTranslationDof::Ux
        })
        .map(|dof| TranslationDofValue {
            dof,
            value: PrescribedValue::Constant(Displacement::new(0.0)),
        })
        .collect();
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
    let realized = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    assert_eq!(realized.len(), 2);
    let summary = assemble_structural_system_borrowed(
        &model,
        &realized,
        Some(&mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    // Independent resultant: unit-volume body plus unit-area right traction.
    let applied = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(i, _)| {
            summary.physical_load[summary
                .structural_dof_layout
                .index(i, StructuralDofKind::Ux)
                .unwrap()]
        })
        .sum::<f64>();
    assert!(
        (applied - stress(0.0)[0]).abs() < 1e-10,
        "applied={applied}"
    );
    (model, mesh, summary)
}
