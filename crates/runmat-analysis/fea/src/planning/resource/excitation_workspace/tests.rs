use super::*;
use crate::fixtures::{
    fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind, FixtureId,
};
use runmat_analysis_core::{
    BodyForceDensity, ExcitationCaseSelection, ExcitationDistribution, ExcitationId,
    SignalPrecision,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn model(mesh: &SolverMeshArtifact) -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "density".into(),
            region_id: "solid".into(),
            kind: FixtureExcitationKind::BodyForce([0.0; 3]),
        }],
    );
    let PhysicalSignal::BodyForceDensity(signal) = &mut model.signals[0] else {
        panic!("density");
    };
    signal.shape = SignalShape::TopologyField {
        topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
        location: FieldLocation::Node,
        component_count: 3,
    };
    signal.representation = SignalRepresentation::Constant(SignalValue::Tensor {
        shape: vec![mesh.topology.nodes.len(), 3],
        values: vec![BodyForceDensity::new(1.0); mesh.topology.nodes.len() * 3],
    });
    model.excitations[0].excitation_mut().distribution = ExcitationDistribution::TopologyField;
    model
}

#[test]
fn field_workspace_counts_actions_not_unique_signals_and_respects_selection() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut model = model(&mesh);
        let first =
            derive_excitation_workspace(&model, &model.sequence.steps[0], Some(&mesh)).unwrap();
        assert_eq!(first.active_field_count, Count(1));
        assert_eq!(
            first.scalar_component_count,
            Count(mesh.topology.nodes.len() as u64 * 3)
        );
        assert!(first.resident_bytes_upper_bound.0 > first.scalar_component_count.0 * 64);
        let mut action = model.excitations[0].clone();
        action.excitation_mut().excitation_id = ExcitationId("other-density".into());
        model.excitation_cases[0]
            .action_ids
            .push(action.excitation().excitation_id.clone());
        model.excitations.push(action);
        let two =
            derive_excitation_workspace(&model, &model.sequence.steps[0], Some(&mesh)).unwrap();
        assert_eq!(two.active_field_count, Count(2));
        assert_eq!(
            two.scalar_component_count.0,
            first.scalar_component_count.0 * 2
        );
        assert_eq!(
            two.resident_bytes_upper_bound.0 - first.resident_bytes_upper_bound.0,
            first.scalar_component_count.0 * 16 + 128
        );
        model.sequence.steps[0].excitation_case_selection = ExcitationCaseSelection::Only(vec![]);
        assert_eq!(
            derive_excitation_workspace(&model, &model.sequence.steps[0], None).unwrap(),
            ExcitationWorkspaceResourceFacts::default()
        );
    }
}

#[test]
fn field_workspace_rejects_foreign_missing_and_wrong_rows_before_sampling() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = model(&mesh);
    assert!(derive_excitation_workspace(&model, &model.sequence.steps[0], None).is_err());
    let PhysicalSignal::BodyForceDensity(signal) = &mut model.signals[0] else {
        panic!("density");
    };
    signal.shape = SignalShape::TopologyField {
        topology_id: Digest::sha256("foreign").to_string(),
        location: FieldLocation::Node,
        component_count: 3,
    };
    assert!(derive_excitation_workspace(&model, &model.sequence.steps[0], Some(&mesh)).is_err());
    let PhysicalSignal::BodyForceDensity(signal) = &mut model.signals[0] else {
        panic!("density");
    };
    signal.shape = SignalShape::TopologyField {
        topology_id: Digest::from_bytes(*mesh.canonical_digest.bytes()).to_string(),
        location: FieldLocation::Node,
        component_count: 3,
    };
    signal.representation = SignalRepresentation::Constant(SignalValue::Tensor {
        shape: vec![4, 3],
        values: vec![BodyForceDensity::new(1.0); 11],
    });
    assert!(derive_excitation_workspace(&model, &model.sequence.steps[0], Some(&mesh)).is_err());
}

#[test]
fn workspace_covers_complex_interpolation_without_a_schedule_multiplier() {
    use runmat_analysis_core::{
        SignalExtrapolation, SignalInterpolation, SignalKnot, TabulatedSignal,
        TabulatedSignalDomain,
    };
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = model(&mesh);
    let real = derive_excitation_workspace(&model, &model.sequence.steps[0], Some(&mesh)).unwrap();
    let PhysicalSignal::BodyForceDensity(signal) = &mut model.signals[0] else {
        panic!("density");
    };
    let SignalRepresentation::Constant(value) = &signal.representation else {
        panic!("constant");
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(
                0.0,
                SignalOrdinate::Complex {
                    real: value.clone(),
                    imaginary: value.clone(),
                },
            ),
            SignalKnot::continuous(
                1.0,
                SignalOrdinate::Complex {
                    real: value.clone(),
                    imaginary: value.clone(),
                },
            ),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Zero,
    });
    signal.precision = SignalPrecision::Complex64;
    assert_eq!(
        derive_excitation_workspace(&model, &model.sequence.steps[0], Some(&mesh)).unwrap(),
        real
    );
}

#[test]
fn workspace_arithmetic_rejects_overflow() {
    assert!(accumulate(
        &mut ExcitationWorkspaceResourceFacts::default(),
        &mut 0,
        &mut 0,
        u64::MAX,
        0
    )
    .is_err());
}

#[test]
fn every_offset_and_later_complex_jump_is_checked_without_sampling() {
    use runmat_analysis_core::{
        SignalExtrapolation, SignalInterpolation, SignalKnot, SignalPointContinuity,
        StandardWaveform, StandardWaveformKind, TabulatedSignal, TabulatedSignalDomain,
    };
    let value = || SignalValue::Tensor {
        shape: vec![4, 3],
        values: vec![1.0; 12],
    };
    let invalid = || SignalValue::Tensor {
        shape: vec![4, 3],
        values: vec![1.0; 11],
    };
    let waveform = SignalRepresentation::Waveform(StandardWaveform {
        kind: StandardWaveformKind::Sine,
        amplitude: value(),
        offset: Some(invalid()),
        start_seconds: runmat_types::physical::Time::new(0.0),
        duration_seconds: (None).map(runmat_types::physical::Time::new),
        frequency_hz: (Some(1.0)).map(runmat_types::physical::Frequency::new),
        phase_radians: runmat_types::physical::Angle::new(0.0),
        duty_cycle: None,
    });
    assert!(validate_representation(&waveform, 4).is_err());
    let table = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: vec![
            SignalKnot::continuous(0.0, SignalOrdinate::Real(value())),
            SignalKnot::jump(
                1.0,
                SignalOrdinate::Real(value()),
                SignalOrdinate::Complex {
                    real: value(),
                    imaginary: invalid(),
                },
                SignalPointContinuity::Right,
            ),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::Zero,
    });
    assert!(validate_representation(&table, 4).is_err());
}

#[test]
fn unresolved_field_artifacts_and_unsupported_locations_cannot_claim_workspace_admission() {
    use runmat_analysis_core::{SignalArtifactDomain, SignalArtifactRef};
    let artifact = SignalRepresentation::<f64>::Artifact(SignalArtifactRef {
        artifact_id: "field".into(),
        content_digest: "sealed".into(),
        schema: "field".into(),
        domain: SignalArtifactDomain::Constant,
        topology_digest: None,
        sample_count: 1,
    });
    assert!(validate_representation(&artifact, 4).is_err());
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut model = model(&mesh);
    let PhysicalSignal::BodyForceDensity(signal) = &mut model.signals[0] else {
        panic!("density");
    };
    let SignalShape::TopologyField { location, .. } = &mut signal.shape else {
        panic!("field");
    };
    *location = FieldLocation::Face;
    assert!(derive_excitation_workspace(&model, &model.sequence.steps[0], Some(&mesh)).is_err());
}
