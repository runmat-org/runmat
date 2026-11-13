use runmat_analysis_core::*;
pub(crate) fn authored_model(current: f64) -> AnalysisModel {
    let mut input =
        crate::physics::electrical::tests::model(StaticElectricalEquation::Electrostatics);
    input.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
        formulation: MagnetostaticFormulation::MagneticVectorPotential,
    });
    for boundary in ["slab_left", "slab_right", "slab_insulated"] {
        input
            .constraints
            .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                constraint_id: boundary.into(),
                kind: MagneticConstraintKind::VectorPotentialGround {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(&(boundary)),
                    ),
                },
            }));
    }
    input
        .constraints
        .push(AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "gauge".into(),
            kind: MagneticConstraintKind::CoulombGauge {
                target: RegionTarget::volume(runmat_meshing_core::fixtures::fixture_solid("slab")),
            },
        }));
    input
        .signals
        .push(PhysicalSignal::ElectricCurrentDensity(Signal {
            signal_id: SignalId("J".into()),
            shape: SignalShape::Vector3,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Vector3(
                [0.0, 0.0, current].map(ElectricCurrentDensity::new),
            )),
            precision: SignalPrecision::Real64,
            provenance: SignalProvenance::Authored,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
        }));
    input
        .excitations
        .push(ExcitationAction::MagneticVolumeCurrentDensity(
            AnalysisExcitation {
                excitation_id: ExcitationId("coil".into()),
                target: ExcitationTarget::WholeModel,
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::IntensityPerVolume,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("J".into()),
                active_step_ids: vec!["potential".into()],
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: false,
                    preserve_moment: false,
                    relative_tolerance: 1e-12,
                },
            },
        ));
    input.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("coil".into()),
        action_ids: vec![ExcitationId("coil".into())],
        activation: ExcitationCaseActivation::Always,
    });
    input
}
