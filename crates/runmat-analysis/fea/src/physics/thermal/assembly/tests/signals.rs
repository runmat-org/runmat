use super::*;

fn temperature_signal(signal_id: &str, values: [f64; 2]) -> PhysicalSignal {
    PhysicalSignal::Temperature(Signal {
        signal_id: SignalId(signal_id.into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: [0.0, 1.0]
                .into_iter()
                .zip(values)
                .map(|(abscissa, value)| {
                    SignalKnot::continuous(
                        abscissa,
                        SignalOrdinate::Real(SignalValue::Scalar(Temperature::new(value))),
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: SignalProvenance::Authored,
    })
}

fn heat_flux_signal(signal_id: &str, values: [f64; 2]) -> PhysicalSignal {
    PhysicalSignal::HeatFlux(Signal {
        signal_id: SignalId(signal_id.into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: [0.0, 1.0]
                .into_iter()
                .zip(values)
                .map(|(abscissa, value)| {
                    SignalKnot::continuous(
                        abscissa,
                        SignalOrdinate::Real(SignalValue::Scalar(HeatFlux::new(value))),
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: SignalProvenance::Authored,
    })
}

// witness: FEA-THERMAL-INPUT-001/boundaries @ thermal_transient/tet4/numerical_kernel;
//   boundary: Tet4 assembly samples prescribed temperature, flux and ambient-time signals; no
//   generated/field-artifact equivalence credit.
#[test]
fn time_signals_drive_prescribed_temperature_flux_and_ambient_temperature() {
    let (mut model, mesh, _, base_region) = thermal_reference_model();
    let flux_region = mesh
        .topology
        .boundary_faces
        .iter()
        .flat_map(|face| &face.provenance)
        .find(|region| region.source_topology_id != base_region)
        .expect("reference mesh has a second boundary region")
        .source_topology_id
        .clone();
    model.signals = vec![
        temperature_signal("prescribed_temperature", [300.0, 320.0]),
        heat_flux_signal("boundary_flux", [100.0, 200.0]),
        temperature_signal("ambient_temperature", [280.0, 340.0]),
    ];
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "time_prescribed_temperature".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(base_region)),
                ),
                value: PrescribedValue::Signal(SignalId("prescribed_temperature".into())),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "time_heat_flux".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_boundary_selection(
            &(flux_region.clone()),
        )),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Signal(SignalId("boundary_flux".into())),
        },
    });

    let at_start = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::time(0.0),
    )
    .unwrap();
    let at_end = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::time(1.0),
    )
    .unwrap();
    assert!(at_start
        .prescribed_temperature_k
        .canonical_values()
        .iter()
        .flatten()
        .all(|value| *value == 300.0));
    assert!(at_end
        .prescribed_temperature_k
        .canonical_values()
        .iter()
        .flatten()
        .all(|value| *value == 320.0));
    let start_flux_power = at_start
        .applied_source
        .canonical_values()
        .iter()
        .sum::<f64>();
    let end_flux_power = at_end.applied_source.canonical_values().iter().sum::<f64>();
    assert!((end_flux_power - 2.0 * start_flux_power).abs() <= 1.0e-12);

    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "time_convection".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_boundary_selection(
            &(flux_region),
        )),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Signal(SignalId("ambient_temperature".into())),
            coefficient: HeatTransferCoefficient::new(5.0),
        },
    }];
    let cold = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::time(0.0),
    )
    .unwrap();
    let hot = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::time(1.0),
    )
    .unwrap();
    let reference = vec![0.0; cold.projection.nodes.len()];
    let cold_ambient_power = cold
        .relative_source(&model, ExcitationSamplePoint::time(0.0), &reference)
        .unwrap()
        .iter()
        .sum::<f64>();
    let hot_ambient_power = hot
        .relative_source(&model, ExcitationSamplePoint::time(1.0), &reference)
        .unwrap()
        .iter()
        .sum::<f64>();
    assert!((hot_ambient_power / cold_ambient_power - 340.0 / 280.0).abs() <= 1.0e-12);
    assert_eq!(cold.conductivity, hot.conductivity);
}

#[test]
fn energy_recovery_uses_the_same_robin_signal_side_as_assembly() {
    use runmat_analysis_core::{SignalPointContinuity, SignalSampleApproach};
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let face = &projection.boundary_faces[0];
    let selector = face.provenance[0].analysis_selection_id().unwrap();
    let ordinate = |value| SignalOrdinate::Real(SignalValue::Scalar(Temperature::new(value)));
    let mut signal = temperature_signal("jump_ambient", [280.0, 340.0]);
    let PhysicalSignal::Temperature(temperature) = &mut signal else {
        unreachable!()
    };
    let SignalRepresentation::Tabulated(table) = &mut temperature.representation else {
        unreachable!()
    };
    table.knots.insert(
        1,
        SignalKnot::jump(
            0.5,
            ordinate(280.0),
            ordinate(340.0),
            SignalPointContinuity::Right,
        ),
    );
    temperature.evidence.sample_count = 3;
    model.signals.push(signal);
    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "jump_convection".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_boundary_selection(
            &(selector),
        )),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Signal(SignalId("jump_ambient".into())),
            coefficient: HeatTransferCoefficient::new(5.0),
        },
    }];
    for (approach, ambient) in [
        (SignalSampleApproach::LeftLimit, 280.0),
        (SignalSampleApproach::At, 340.0),
        (SignalSampleApproach::RightLimit, 340.0),
    ] {
        let sample = ExcitationSamplePoint::TimeSeconds {
            value: 0.5,
            approach,
        };
        let assembly = assemble(
            &model,
            &mesh,
            crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
                .unwrap()
                .support(),
            &RealizedExcitations::default(),
            sample,
        )
        .unwrap();
        let equilibrium = vec![ambient; projection.nodes.len()];
        let energy = assembly
            .energy_balance(&model, sample, &equilibrium, &equilibrium, 1.0, 0.0)
            .unwrap();
        assert_eq!(
            energy.boundary_loss_w.canonical_value(),
            0.0,
            "{approach:?}"
        );
        assert_eq!(energy.applied_power_w.canonical_value(), 0.0);
        assert_eq!(energy.relative_residual, 0.0);
    }
}
