use super::*;
use runmat_analysis_core::{
    ElectricChargeDensity, ExcitationAction, ExcitationDistribution, PhysicalQuantityKind,
    PhysicalSignal, Signal, SignalId, SignalOrdinate, SignalPrecision, SignalRepresentation,
    SignalShape, SignalValue,
};

fn charge_model(
    representation: SignalRepresentation<ElectricChargeDensity>,
) -> runmat_analysis_core::AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let id = SignalId("volume.charge_density".into());
    let evidence = match &model.signals[0] {
        PhysicalSignal::Force(signal) => signal.evidence.clone(),
        _ => unreachable!(),
    };
    model.signals = vec![PhysicalSignal::ElectricChargeDensity(Signal {
        signal_id: id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation,
        precision: SignalPrecision::Real64,
        evidence,
        provenance: runmat_analysis_core::SignalProvenance::Authored,
    })];
    let mut action = model.excitations[0].excitation().clone();
    action.signal_id = id;
    action.target = runmat_analysis_core::ExcitationTarget::WholeModel;
    action.direction = None;
    action.distribution = ExcitationDistribution::IntensityPerVolume;
    action.projection.preserve_resultant = true;
    action.projection.preserve_moment = false;
    model.excitations = vec![ExcitationAction::ElectricChargeDensity(action)];
    model
}

fn scalar(value: f64) -> SignalValue<ElectricChargeDensity> {
    SignalValue::Scalar(ElectricChargeDensity::new(value))
}

#[test]
fn realizes_signed_static_volume_charge_without_reinterpreting_total_charge() {
    for density in [-2.0, 0.0, 2.0] {
        let model = charge_model(SignalRepresentation::Constant(scalar(density)));
        let realized = realize_static(&model).unwrap();
        let load = realized.iter().next().unwrap();
        assert_eq!(load.quantity, PhysicalQuantityKind::ElectricChargeDensity);
        assert_eq!(
            load.policy.distribution,
            ExcitationDistribution::IntensityPerVolume
        );
        assert!(matches!(
            load.target,
            super::super::RealizedExcitationTarget::WholeModel
        ));
        assert!(
            matches!(load.kind, RealizedExcitationKind::ElectricChargeDensity(value)
            if value.real == density && value.imaginary == 0.0)
        );
        assert!(!load.kind.is_structural_load());
    }
    let mut model = charge_model(SignalRepresentation::Constant(scalar(2.0)));
    model.excitations[0] =
        ExcitationAction::ElectricCharge(model.excitations[0].excitation().clone());
    assert_eq!(
        realize_static(&model).unwrap_err().to_string(),
        "excitation action and physical signal quantity do not match"
    );
}

#[test]
fn charge_density_rejects_vector_values_and_ambiguous_static_tables() {
    let vector = SignalValue::Vector3([ElectricChargeDensity::new(1.0); 3]);
    let model = charge_model(SignalRepresentation::Constant(vector));
    assert!(realize_static(&model).is_err());
    let model = charge_model(SignalRepresentation::Tabulated(
        runmat_analysis_core::TabulatedSignal {
            domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                runmat_analysis_core::SignalKnot::continuous(
                    0.0,
                    SignalOrdinate::Real(scalar(1.0)),
                ),
                runmat_analysis_core::SignalKnot::continuous(
                    1.0,
                    SignalOrdinate::Real(scalar(2.0)),
                ),
            ],
            interpolation: runmat_analysis_core::SignalInterpolation::Linear,
            extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
        },
    ));
    assert_eq!(
        realize_static(&model).unwrap_err().to_string(),
        "static excitation requires constant tabulated values"
    );
    let realized = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::time(0.5),
    )
    .unwrap();
    assert!(matches!(realized.iter().next().unwrap().kind,
        RealizedExcitationKind::ElectricChargeDensity(value) if value.real == 1.5));
}

#[test]
fn preserves_complex_charge_density_components_for_shared_frequency_sampling() {
    let mut model = charge_model(SignalRepresentation::Tabulated(
        runmat_analysis_core::TabulatedSignal {
            domain: runmat_analysis_core::TabulatedSignalDomain::FrequencyHertz,
            knots: vec![runmat_analysis_core::SignalKnot::continuous(
                60.0,
                SignalOrdinate::Complex {
                    real: scalar(2.0),
                    imaginary: scalar(-3.0),
                },
            )],
            interpolation: runmat_analysis_core::SignalInterpolation::Linear,
            extrapolation: runmat_analysis_core::SignalExtrapolation::HoldEndpoint,
        },
    ));
    let PhysicalSignal::ElectricChargeDensity(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    signal.precision = SignalPrecision::Complex64;
    let realized = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::frequency(60.0),
    )
    .unwrap();
    assert!(matches!(realized.iter().next().unwrap().kind,
        RealizedExcitationKind::ElectricChargeDensity(value)
        if value.real == 2.0 && value.imaginary == -3.0));
    for (component, expected) in [
        (super::super::ComplexComponent::Real, 2.0),
        (super::super::ComplexComponent::Imaginary, -3.0),
    ] {
        let projected = realized.component(component);
        assert!(matches!(projected.iter().next().unwrap().kind,
            RealizedExcitationKind::ElectricChargeDensity(value)
            if value.real == expected && value.imaginary == 0.0));
    }
}

#[test]
fn scalar_volume_projection_preserves_signed_charge_and_detects_loss() {
    use crate::excitation::{
        ScalarProjectionAccumulator, SpatialIntegrationRule, SpatialProjectionBalance,
    };
    use crate::topology::AnalysisProjection;
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    // Exercises shared evidence, not the solver's independent element quadrature tests.
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        for density in [-2.0, 0.0, 2.0] {
            let model = charge_model(SignalRepresentation::Constant(scalar(density)));
            let realized = realize_static(&model).unwrap();
            let load = realized.iter().next().unwrap();
            for fraction in [1.0, 0.5] {
                let mut accumulator = ScalarProjectionAccumulator::new(
                    load,
                    &projection,
                    SpatialIntegrationRule::VolumeKeastFifteen,
                    1,
                )
                .unwrap();
                accumulator.add_requested(density, 0.0);
                accumulator.add_projected(0, density * fraction, 0.0);
                let evidence = accumulator.finish();
                evidence.validate().unwrap();
                assert_eq!(
                    evidence.quantity,
                    PhysicalQuantityKind::ElectricChargeDensity
                );
                assert_eq!(evidence.accepted, density == 0.0 || fraction == 1.0);
                let SpatialProjectionBalance::ScalarIntegral {
                    requested_integral,
                    projected_integral,
                    ..
                } = evidence.balance
                else {
                    panic!("scalar charge balance")
                };
                assert_eq!(requested_integral.real, density);
                assert_eq!(projected_integral.real, density * fraction);
            }
        }
    }
}
