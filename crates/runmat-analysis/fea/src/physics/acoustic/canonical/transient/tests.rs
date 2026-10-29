//! Independent d'Alembert references for a pressure pulse in a unit duct (c = 2 m/s, rho = 3).
use super::*;
use crate::topology::test_meshes::quadratic_slab_mesh;
use runmat_analysis_core::{
    AcousticConstraint, AnalysisBoundary, BoundaryOperator, PhysicalSignal, PrescribedValue,
    PropertyRegionTarget, RegionTarget, Signal, SignalApproximationEvidence, SignalExtrapolation,
    SignalId, SignalInterpolation, SignalKnot, SignalOrdinate, SignalPrecision, SignalProvenance,
    SignalRepresentation, SignalShape, SignalValue, SpecificAcousticImpedance, TabulatedSignal,
    TabulatedSignalDomain,
};

/// Triangular inlet pulse: 0 -> 1 Pa at 0.2 s -> 0 at 0.4 s, then zero (0.8 m long).
fn pulse(t: f64) -> f64 {
    if t <= 0.0 || t >= 0.4 {
        0.0
    } else if t <= 0.2 {
        t / 0.2
    } else {
        (0.4 - t) / 0.2
    }
}

fn pulse_signal() -> PhysicalSignal {
    PhysicalSignal::Pressure(Signal {
        signal_id: SignalId("pulse".into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: [(0.0, 0.0), (0.2, 1.0), (0.4, 0.0), (10.0, 0.0)]
                .into_iter()
                .map(|(t, value)| {
                    SignalKnot::continuous(
                        t,
                        SignalOrdinate::Real(SignalValue::Scalar(Pressure::new(value))),
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
            sample_count: 4,
        },
        provenance: SignalProvenance::Authored,
    })
}

use runmat_analysis_core::Pressure;

fn duct(end: Option<f64>, end_time: f64, intervals: usize) -> AnalysisModel {
    let mut model = super::super::tests::model();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.sequence.steps = vec![ProcedureStep::new(
        "acoustic",
        PhysicsFamily::Acoustic,
        AnalysisProcedure::TransientResponse,
        SolutionSpace::Direct,
        IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: end_time,
            interval_count: intervals,
        },
        Vec::new(),
    )];
    model.signals.push(pulse_signal());
    model
        .constraints
        .push(AnalysisConstraint::Acoustic(AcousticConstraint {
            constraint_id: "drive".into(),
            kind: AcousticConstraintKind::PrescribedPressure {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Signal(SignalId("pulse".into())),
            },
        }));
    if let Some(impedance) = end {
        model.boundaries.push(AnalysisBoundary {
            boundary_id: "termination".into(),
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "slab_right",
            )),
            operator: BoundaryOperator::AcousticImpedance {
                impedance: SpecificAcousticImpedance::new(impedance),
            },
        });
    }
    model
}

fn line_error(
    result: &CanonicalAcousticTransient,
    index: usize,
    reference: impl Fn(f64) -> f64,
) -> f64 {
    let values = &result.pressure_pa[index];
    let squared = result
        .projection
        .nodes
        .iter()
        .zip(values)
        .map(|(node, value)| (value - reference(node.coordinates_m.canonical_value()[0])).powi(2))
        .sum::<f64>();
    (squared / values.len() as f64).sqrt()
}

#[test]
fn matched_duct_pulse_travels_at_sound_speed_without_reflection() {
    // p(x,t) = g(t - x/c) with a reflection-free rho c termination.
    let mut errors = Vec::new();
    for (cells, intervals) in [(8, 480), (12, 480)] {
        let result = solve_canonical_transient(
            &duct(Some(6.0), 1.2, intervals),
            &quadratic_slab_mesh(cells),
            "acoustic",
            AcousticTransientPolicy::default(),
            &TransientRetention::Every,
        )
        .unwrap();
        let index = result
            .time_points_s
            .iter()
            .position(|t| (t - 0.6).abs() < 1e-12)
            .unwrap();
        errors.push(line_error(&result, index, |x| pulse(0.6 - x / 2.0)));
        // Arrival at the outlet: peak at t = 0.2 + L/c = 0.7 s.
        let outlet = result
            .projection
            .nodes
            .iter()
            .position(|node| (node.coordinates_m.canonical_value()[0] - 1.0).abs() < 1e-12)
            .unwrap();
        let (peak_index, peak) = result
            .pressure_pa
            .iter()
            .map(|values| values[outlet])
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let arrival = result.time_points_s[peak_index];
        assert!(
            (arrival - 0.7).abs() <= 0.0125,
            "cells={cells} arrival {arrival}"
        );
        assert!((peak - 1.0).abs() < 0.1, "cells={cells} peak {peak}");
        // After the pulse leaves, almost nothing is reflected back into the duct.
        let last = result.pressure_pa.last().unwrap();
        assert!(
            last.iter().all(|value| value.abs() < 0.05),
            "residual reflection"
        );
        // The termination absorbs what the inlet put in: energy stays bounded.
        assert!(result.boundary_absorbed_energy_j.last().unwrap() > &0.0);
    }
    assert!(errors[1] < errors[0] && errors[1] < 0.05, "{errors:?}");
}

#[test]
fn rigid_end_doubles_the_pulse_and_conserves_energy_once_the_drive_stops() {
    let result = solve_canonical_transient(
        &duct(None, 1.2, 480),
        &quadratic_slab_mesh(8),
        "acoustic",
        AcousticTransientPolicy::default(),
        &TransientRetention::Every,
    )
    .unwrap();
    let outlet = result
        .projection
        .nodes
        .iter()
        .position(|node| (node.coordinates_m.canonical_value()[0] - 1.0).abs() < 1e-12)
        .unwrap();
    let peak = result
        .pressure_pa
        .iter()
        .map(|values| values[outlet])
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (peak - 2.0).abs() < 0.15,
        "rigid-wall pressure doubling {peak}"
    );
    // The inlet is held at zero after 0.4 s: no work, no absorption, energy is constant once
    // the imposed-history differences of the inlet kinematics (two steps) have cleared.
    let start = result
        .time_points_s
        .iter()
        .position(|t| *t >= 0.4 + 1e-12)
        .unwrap()
        + 3;
    let reference = result.acoustic_energy_j[start];
    for energy in &result.acoustic_energy_j[start..] {
        assert!(
            ((energy - reference) / reference).abs() < 1e-8,
            "{energy} vs {reference}"
        );
    }
}
