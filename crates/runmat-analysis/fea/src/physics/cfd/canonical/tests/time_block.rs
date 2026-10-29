//! Independent acceleration and exact original-field temporal block composition.
use super::super::geometry_ops::*;
use super::{closed_reference::closed, time_signals::table, *};
use crate::execution::{FluidTransientBlock, FluidTransientBlockResult, FluidTransientSchedule};
use runmat_analysis_core::*;

fn block_model(f: &fixture::Fixture) -> AnalysisModel {
    let (mut workload, _) = crate::execution::fluid_test_workload();
    workload.model.constraints = f
        .constraints
        .iter()
        .cloned()
        .map(AnalysisConstraint::Fluid)
        .collect();
    workload.model.boundaries = f.operators.clone();
    workload.model.signals = f.signals.clone();
    workload.model.cfd = Some(f.material.clone());
    let mut step = ProcedureStep::fluid_transient("advance");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.,
        start: 0.,
        end: 0.3,
        interval_count: 3,
    };
    workload.model.sequence = AnalysisSequence::from_steps("original-fluid-blocks", vec![step]);
    workload.model
}

// witness: FEA-FLUID-TRANSIENT-001/history @
//   incompressible_laminar_transient/canonical_tetrahedra/numerical_kernel;
//   boundary: Original acceleration history and split replay; chronology/tamper/budget/cancel
//   checks, no native process credit.
#[test]
fn authored_time_block_cold_recovery_and_split_match_original_acceleration() {
    let start = [0.001, 0.002, -0.001];
    let acceleration = [0.01, -0.005, 0.002];
    let end: [f64; 3] = std::array::from_fn(|k| start[k] + acceleration[k]);
    let mut f = closed(2, start, 0.0);
    f.signals = vec![PhysicalSignal::Velocity(table(
        "u",
        SignalShape::Vector3,
        Some(CoordinateFrame::Global),
        [start, end].map(|v| SignalValue::Vector3(v.map(Velocity::new))),
    ))];
    for c in &mut f.constraints {
        if let FluidConstraintKind::PrescribedVelocity { value, .. } = &mut c.kind {
            *value = PrescribedValue::Signal(SignalId("u".into()));
        }
    }
    let model = block_model(&f);
    let step = &model.sequence.steps[0];
    let u = vec![start; f.topology.cells.len()];
    let zero = vec![0.; u.len()];
    let q = f
        .topology
        .faces
        .iter()
        .map(|face| dot(start, face.owner_area_vector_m2))
        .collect::<Vec<_>>();
    let previous = CanonicalFluidFields {
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(&u),
        pressure_reference_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(&zero),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(&zero),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(&q),
    };
    let full =
        FluidTransientBlock::prepare(&model, &step.step_id, f.input(), previous, 0..3).unwrap();
    assert_eq!(
        full.schedule().points_s(),
        &[0., 0.09999999999999999, 0.19999999999999998, 0.3]
    );
    let result = full.execute().unwrap();
    // Without sampled outputs the block retains its terminal and that interval's predecessor;
    // every interval keeps its evidence.
    assert_eq!(result.intervals().len(), 3);
    assert_eq!(
        result
            .retained_results()
            .iter()
            .map(|(o, _)| *o)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    for (ordinal, r) in result.retained_results() {
        let t = full.schedule().points_s()[ordinal + 1];
        let expected: [f64; 3] = std::array::from_fn(|k| start[k] + t * acceleration[k]);
        for v in r.velocity_m_per_s.canonical_values() {
            assert!(norm(sub(*v, expected)) < 1e-8);
        }
        let p = CanonicalFluidProblem::prepare_at(
            f.input(),
            crate::excitation::ProjectionSample::TimeSeconds {
                value: t,
                approach: SignalSampleApproach::At,
            },
        )
        .unwrap();
        let gradients = super::super::pressure_gradient::evaluate(
            &p,
            r.pressure_fluctuation_pa.canonical_values(),
        )
        .unwrap();
        for g in gradients {
            assert!(norm(add(g, acceleration)) < 2e-7);
        }
    }
    let bytes = result.canonical_bytes().unwrap();
    assert!((bytes.len() as u64) <= full.reservation().result_encoded_bytes_upper_bound);
    let cold = FluidTransientBlockResult::from_canonical_bytes(&bytes).unwrap();
    full.recover(&cold).unwrap();
    let first = FluidTransientBlock::prepare(&model, &step.step_id, f.input(), previous, 0..1)
        .unwrap()
        .execute()
        .unwrap();
    let latter = FluidTransientBlock::prepare(
        &model,
        &step.step_id,
        f.input(),
        first.terminal().unwrap().original_fields(),
        1..3,
    )
    .unwrap();
    let resumed = latter.execute().unwrap();
    latter.recover(&resumed).unwrap();
    assert_eq!(resumed.intervals(), &result.intervals()[1..]);
    assert_eq!(resumed.terminal(), result.terminal());
    let mut tampered = serde_json::to_value(&cold).unwrap();
    tampered["retained"][1][1]["pressure_reference_pa"][0] = serde_json::json!(2.);
    let tampered: FluidTransientBlockResult = serde_json::from_value(tampered).unwrap();
    assert!(full.recover(&tampered).is_err());
    let mut reordered = serde_json::to_value(&cold).unwrap();
    reordered["retained"].as_array_mut().unwrap().swap(0, 1);
    assert!(full
        .recover(&serde_json::from_value(reordered).unwrap())
        .is_err());
    let mut wrong = u.clone();
    wrong[0][0] += 0.1;
    let wrong = CanonicalFluidFields {
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(&wrong),
        ..previous
    };
    assert!(
        FluidTransientBlock::prepare(&model, &step.step_id, f.input(), wrong, 0..3)
            .unwrap()
            .recover(&cold)
            .is_err()
    );
    let mut low = f.input();
    low.policy.maximum_operator_work = Some(full.reservation().operator_work_upper_bound - 1);
    assert!(FluidTransientBlock::prepare(&model, &step.step_id, low, previous, 0..3).is_err());
    assert!(
        FluidTransientBlock::prepare(&model, &step.step_id, f.input(), previous, 0..4).is_err()
    );
    let cancellation = crate::progress::FeaCancellationToken::new();
    cancellation.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation,
            ..Default::default()
        });
    assert!(full.execute().is_err());
}

#[test]
fn fluid_time_schedule_uses_authored_observations_and_rejects_false_step_domain() {
    let (mut w, _) = crate::execution::fluid_test_workload();
    let mut s = ProcedureStep::fluid_transient("fluid");
    s.independent_domain = IndependentDomain::TimeSeconds {
        origin: 2.,
        start: 2.,
        end: 3.,
        interval_count: 2,
    };
    w.model.sequence.steps = vec![s.clone()];
    let schedule = FluidTransientSchedule::derive(&w.model, &s).unwrap();
    assert_eq!(schedule.points_s(), &[2., 2.5, 3.]);
    assert_eq!(schedule.arrival(3), None);
    s.accuracy.resources.maximum_samples = 2;
    w.model.sequence.steps[0] = s.clone();
    assert!(FluidTransientSchedule::derive(&w.model, &s).is_err());
    let mut tiny = w.model.clone();
    tiny.sequence.steps[0].accuracy.resources.maximum_samples = 3;
    tiny.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(FluidTransientSchedule::derive(&tiny, &tiny.sequence.steps[0]).is_err());
    s.independent_domain = IndependentDomain::Static;
    w.model.sequence.steps[0] = s.clone();
    assert!(FluidTransientSchedule::derive(&w.model, &s).is_err());
    w.model.sequence.steps[0].family = PhysicsFamily::Structural;
    assert!(FluidTransientSchedule::derive(&w.model, &w.model.sequence.steps[0]).is_err());
}

#[test]
fn gauge_jump_arrival_and_requested_time_survive_original_field_blocks() {
    let velocity = [0.001, 0.002, -0.001];
    let mut f = closed(2, velocity, 0.);
    let mut gauge = table(
        "gauge-event",
        SignalShape::Scalar,
        None,
        [0., 1e16].map(|v| SignalValue::Scalar(Pressure::new(v))),
    );
    let ordinate = |v| SignalOrdinate::Real(SignalValue::Scalar(Pressure::new(v)));
    let SignalRepresentation::Tabulated(t) = &mut gauge.representation else {
        unreachable!()
    };
    t.knots = vec![
        SignalKnot::continuous(0., ordinate(0.)),
        SignalKnot::jump(
            0.15,
            ordinate(0.),
            ordinate(1e16),
            SignalPointContinuity::Right,
        ),
        SignalKnot::continuous(1., ordinate(1e16)),
    ];
    gauge.evidence.sample_count = 3;
    f.signals.push(PhysicalSignal::Pressure(gauge));
    let FluidConstraintKind::PressureReference { value, .. } =
        &mut f.constraints.last_mut().unwrap().kind
    else {
        unreachable!()
    };
    *value = PrescribedValue::Signal(SignalId("gauge-event".into()));
    let mut model = block_model(&f);
    let step = &mut model.sequence.steps[0];
    step.observations
        .push(ObservationRequest::History(ObservationSpec {
            observation_id: ObservationId("physical-history".into()),
            quantity: ObservationQuantity::FluidVelocity,
            value_shape: ObservationValueShape::Vector3,
            target: ObservationTarget::WholeModel,
            source_location: FieldLocation::Cell,
            output_location: FieldLocation::Cell,
            recovery: RecoveryPolicy::Native,
            component: ObservationComponent::All,
            frame: CoordinateFrame::Global,
            sign_convention: SignConvention::Native,
            complex_representation: ComplexRepresentation::Real,
            sampling: SamplingPolicy::TimeSeconds(vec![0.225]),
            reduction: None,
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
                operation_count: 100000,
                peak_resident_bytes: 1000000,
                stored_bytes: 1000000,
                transfer_bytes: 1000000,
                collective_count: 0,
            }),
        }));
    let step_id = step.step_id.clone();
    let u = vec![velocity; f.topology.cells.len()];
    let zero = vec![0.; u.len()];
    let q = f
        .topology
        .faces
        .iter()
        .map(|face| dot(velocity, face.owner_area_vector_m2))
        .collect::<Vec<_>>();
    let initial = CanonicalFluidFields {
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(&u),
        pressure_reference_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(&zero),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(&zero),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(&q),
    };
    let block = FluidTransientBlock::prepare(&model, &step_id, f.input(), initial, 0..5).unwrap();
    assert_eq!(
        block.schedule().points_s(),
        &[
            0.,
            0.09999999999999999,
            0.15,
            0.19999999999999998,
            0.225,
            0.3
        ]
    );
    assert_eq!(
        block.schedule().arrival(2),
        Some(SignalSampleApproach::LeftLimit)
    );
    assert_eq!(
        block.schedule().departure(2),
        Some(SignalSampleApproach::RightLimit)
    );
    let result = block.execute().unwrap();
    for (i, r) in result.retained_results() {
        let i = *i;
        for v in r.velocity_m_per_s.canonical_values() {
            assert!(norm(sub(*v, velocity)) < 1e-8);
        }
        let expected = if i < 2 { 0. } else { 1e16 };
        assert!(r
            .pressure_reference_pa
            .canonical_values()
            .iter()
            .all(|v| *v == expected));
        assert!(r
            .pressure_fluctuation_pa
            .canonical_values()
            .iter()
            .all(|v| v.abs() < 1e-7));
    }
    let cold = FluidTransientBlockResult::from_canonical_bytes(&result.canonical_bytes().unwrap())
        .unwrap();
    block.recover(&cold).unwrap();
    let first = FluidTransientBlock::prepare(&model, &step_id, f.input(), initial, 0..2)
        .unwrap()
        .execute()
        .unwrap();
    let resumed = FluidTransientBlock::prepare(
        &model,
        &step_id,
        f.input(),
        first.terminal().unwrap().original_fields(),
        2..5,
    )
    .unwrap()
    .execute()
    .unwrap();
    assert_eq!(resumed.intervals(), &result.intervals()[2..]);
    assert_eq!(resumed.terminal(), result.terminal());
    model.sequence.steps[0].accuracy.resources.maximum_samples = 5;
    assert!(FluidTransientBlock::prepare(&model, &step_id, f.input(), initial, 0..5).is_err());
}
