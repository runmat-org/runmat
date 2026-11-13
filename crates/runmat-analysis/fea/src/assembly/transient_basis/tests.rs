use super::*;
use crate::fixtures::{
    fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind, FixtureId,
};
use runmat_analysis_core::*;
use std::cell::Cell;

thread_local! {
    static OPERATOR_ASSEMBLIES: Cell<usize> = const { Cell::new(0) };
}

pub(in crate::assembly) fn record_operator_assembly() {
    OPERATOR_ASSEMBLIES.with(|count| count.set(count.get() + 1));
}

fn count() -> usize {
    OPERATOR_ASSEMBLIES.with(Cell::get)
}

fn initial(model: &AnalysisModel, mesh: &SolverMeshArtifact) -> AssemblySummary {
    let loads = realize_step_excitations(
        model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::time(0.0),
    )
    .unwrap();
    assemble_structural_system_borrowed(
        model,
        &loads,
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap()
}

fn compare(model: &AnalysisModel) {
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(model);
    let before = count();
    let initial = initial(model, &mesh);
    assert_eq!(count(), before + 1);
    let basis = PreparedStructuralTransientBasis::from_admitted(
        model,
        &model.sequence.steps[0],
        Some(&mesh),
        &initial,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        count(),
        before + 1,
        "preparation must not assemble operators again"
    );
    assert!(std::ptr::eq(basis.initial, &initial));
    assert!(std::ptr::eq(basis.mesh, &mesh));
    let mut previous = None;
    for (time, approach) in [
        (0.0, SignalSampleApproach::At),
        (0.25, SignalSampleApproach::At),
        (0.5, SignalSampleApproach::LeftLimit),
        (0.5, SignalSampleApproach::RightLimit),
        (1.0, SignalSampleApproach::At),
    ] {
        let point = ExcitationSamplePoint::TimeSeconds {
            value: time,
            approach,
        };
        let before_sample = count();
        let sampled = basis.sample(point).unwrap();
        assert_eq!(
            count(),
            before_sample,
            "sampling must perform no element/operator assembly"
        );
        let loads = realize_step_excitations(model, &model.sequence.steps[0], point).unwrap();
        let full = assemble_structural_system_borrowed(
            model,
            &loads,
            Some(&mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        assert_eq!(
            sampled, full,
            "all numerical metadata and sampled provenance must match at {point:?}"
        );
        assert_eq!(sampled.operator.operators(), initial.operator.operators());
        if let Some(previous) = previous {
            assert_ne!(
                sampled.structural_load_plan_identity, previous,
                "equal-valued constant loads still require fresh sample identity"
            );
        }
        previous = Some(sampled.structural_load_plan_identity);
        for evidence in &sampled.spatial_projection_evidence {
            assert_eq!(
                evidence.sample,
                crate::excitation::ProjectionSample::TimeSeconds {
                    value: time,
                    approach
                }
            );
        }
    }
}

#[test]
fn complete_summaries_match_for_constant_force_moment_traction_pressure_and_body_force() {
    for kind in [
        FixtureExcitationKind::Force([1.0, 2.0, 3.0]),
        FixtureExcitationKind::Moment([1.0, 2.0, 3.0]),
        FixtureExcitationKind::Traction([1.0, 2.0, 3.0]),
        FixtureExcitationKind::Pressure(3.0),
        FixtureExcitationKind::BodyForce([1.0, 2.0, 3.0]),
    ] {
        let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "sampled".into(),
                region_id: if matches!(kind, FixtureExcitationKind::BodyForce(_)) {
                    "dynamic_volume"
                } else {
                    "tip"
                }
                .into(),
                kind,
            }],
        );
        compare(&model);
    }
}

#[test]
fn complete_summaries_match_for_varied_and_discontinuous_forcing() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("fixture force");
    };
    let value = |x| {
        SignalOrdinate::Real(SignalValue::Vector3([
            Force::new(x),
            Force::new(0.0),
            Force::new(0.0),
        ]))
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(0.0, value(1.0)),
            SignalKnot::jump(0.5, value(2.0), value(-1.0), SignalPointContinuity::Right),
            SignalKnot::continuous(1.0, value(3.0)),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Reject,
    });
    signal.evidence.sample_count = 3;
    compare(&model);
}

#[test]
fn complete_summaries_match_for_affine_constraints() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let c = std::f64::consts::FRAC_1_SQRT_2;
    // Keep the fully fixed root and tip-z support, but replace the tip-y support
    // with one partial non-axis constraint. Its surviving free-tip row couples x/y
    // instead of reducing to a fully fixed Cartesian span.
    model.constraints[1] = AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "oblique-tip-support".into(),
        frame: CoordinateFrame::Cartesian(CartesianFrame {
            frame_id: "rotated-tip".into(),
            origin: Position::new([0.0; 3]),
            x_axis: [c, c, 0.0],
            y_axis: [-c, c, 0.0],
        }),
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "free_vertex_y_support",
            )),
            translations: vec![TranslationDofValue {
                dof: StructuralTranslationDof::Ux,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            }],
            rotations: Vec::new(),
        }),
    });
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    assert!(initial(&model, &mesh).affine_constraint_reduction.is_some());
    compare(&model);
}

#[test]
fn complete_summaries_match_for_prescribed_motion_constraint_channel() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let acceleration_id = SignalId("support-acceleration".into());
    model.signals.push(PhysicalSignal::Acceleration(Signal {
        signal_id: acceleration_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3(
            [Acceleration::new(1.0); 3],
        )),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }));
    let root = model.constraints[0].clone();
    model.constraints[0] = AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "moving-root-x".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::PrescribedMotion(PrescribedStructuralMotion {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
            translations: vec![TranslationDofMotion {
                dof: StructuralTranslationDof::Ux,
                history: StructuralKinematicHistory::AccelerationDriven {
                    acceleration: AccelerationSignalId(acceleration_id),
                    initial_displacement: Displacement::new(0.0),
                    initial_velocity: Velocity::new(0.0),
                    reconstruction:
                        StructuralMotionReconstruction::PiecewiseLinearAccelerationExact,
                },
            }],
        }),
    });
    let AnalysisConstraint::Structural(mut root) = root else {
        panic!("root");
    };
    root.constraint_id = "fixed-root-yz".into();
    root.kind = StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
        translations: [StructuralTranslationDof::Uy, StructuralTranslationDof::Uz]
            .into_iter()
            .map(|dof| TranslationDofValue {
                dof,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            })
            .collect(),
        rotations: Vec::new(),
    });
    model.constraints.push(AnalysisConstraint::Structural(root));
    compare(&model);
}

#[test]
fn foreign_mesh_material_step_constraints_and_unmapped_sources_do_not_reuse_authority() {
    let model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let summary = initial(&model, &mesh);
    let mut foreign_mesh = mesh.clone();
    {
        let quantity_change = 0.01;
        let quantity_index = 0;
        let quantity_slot = &mut foreign_mesh.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    foreign_mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    foreign_mesh.seal_canonical_digest().unwrap();
    assert!(PreparedStructuralTransientBasis::from_admitted(
        &model,
        &model.sequence.steps[0],
        Some(&foreign_mesh),
        &summary
    )
    .is_err());
    let mut foreign_material = model.clone();
    foreign_material.materials[0].mass_density_kg_per_m3 =
        (Some(1234.0)).map(runmat_types::physical::MassDensity::new);
    assert!(PreparedStructuralTransientBasis::from_admitted(
        &foreign_material,
        &foreign_material.sequence.steps[0],
        Some(&mesh),
        &summary
    )
    .is_err());
    let foreign_step = ProcedureStep::structural_transient("foreign");
    assert!(PreparedStructuralTransientBasis::from_admitted(
        &model,
        &foreign_step,
        Some(&mesh),
        &summary
    )
    .is_err());
    let mut foreign_constraints = model.clone();
    foreign_constraints.constraints.remove(1);
    let basis = PreparedStructuralTransientBasis::from_admitted(
        &foreign_constraints,
        &foreign_constraints.sequence.steps[0],
        Some(&mesh),
        &summary,
    )
    .unwrap()
    .unwrap();
    assert!(basis.sample(ExcitationSamplePoint::time(0.25)).is_err());
    let mut foreign_source = model.clone();
    replace_fixture_excitations(
        &mut foreign_source,
        vec![FixtureExcitation {
            id: "foreign".into(),
            region_id: "not-a-mesh-region".into(),
            kind: FixtureExcitationKind::Force([1.0, 0.0, 0.0]),
        }],
    );
    let basis = PreparedStructuralTransientBasis::from_admitted(
        &foreign_source,
        &foreign_source.sequence.steps[0],
        Some(&mesh),
        &summary,
    )
    .unwrap()
    .unwrap();
    assert!(basis.sample(ExcitationSamplePoint::time(0.25)).is_err());
}

#[test]
fn coupled_explicit_synthetic_and_connector_cases_keep_full_assembly_boundary() {
    let model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let summary = initial(&model, &mesh);
    assert!(PreparedStructuralTransientBasis::from_admitted(
        &model,
        &model.sequence.steps[0],
        None,
        &summary
    )
    .unwrap()
    .is_none());
    let mixed = crate::fixtures::mixed_solid_beam_model([0.0; 3]);
    assert!(PreparedStructuralTransientBasis::from_admitted(
        &mixed,
        &mixed.sequence.steps[0],
        Some(&mesh),
        &summary
    )
    .unwrap()
    .is_none());
    let mut coupled = model.clone();
    coupled.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    assert!(coupled.thermo_mechanical.is_some());
    assert!(PreparedStructuralTransientBasis::from_admitted(
        &coupled,
        &coupled.sequence.steps[0],
        Some(&mesh),
        &summary
    )
    .unwrap()
    .is_none());
    let mut connector = model.clone();
    connector.connectors.push(AnalysisConnector {
        connector_id: "spring".into(),
        kind: ConnectorKind::IsotropicGroundSpring {
            target: RegionTarget::point("node:1"),
            stiffness: Stiffness::new(1.0),
        },
    });
    assert!(PreparedStructuralTransientBasis::from_admitted(
        &connector,
        &connector.sequence.steps[0],
        Some(&mesh),
        &summary
    )
    .unwrap()
    .is_none());
}
