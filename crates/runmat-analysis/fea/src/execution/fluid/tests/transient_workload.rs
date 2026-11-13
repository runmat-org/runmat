//! Source-authenticated initial data and independent original transient references.
use super::*;
use crate::execution::{FluidTransientSolverResult, FluidTransientWorkload, InitialFieldBinding};

fn authored_transient() -> (FluidTransientWorkload, SolverMeshArtifact) {
    let (steady, mesh) = super::closed_reference::closed_authored();
    let mut model = steady.model;
    model.sequence = AnalysisSequence::from_steps(
        "physical-time",
        vec![ProcedureStep::fluid_transient("advance")],
    );
    model.sequence.steps[0].independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.,
        start: 0.,
        end: 0.3,
        interval_count: 3,
    };
    (
        FluidTransientWorkload {
            model,
            step_id: ProcedureStepId("advance".into()),
            policy: CanonicalFluidPolicy {
                linear_relative_tolerance: steady.policy.linear_relative_tolerance,
                ..CanonicalFluidPolicy::default()
            },
            geometry_policy: steady.geometry_policy,
            initial_fields: Vec::new(),
        },
        mesh,
    )
}
fn initial_velocity(value: InitialValue<Velocity>) -> InitialCondition {
    InitialCondition::FluidVelocity(InitialConditionValue {
        initial_condition_id: "original-u".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value,
        applies_before_step_id: "advance".into(),
    })
}
fn ramp() -> Signal<Velocity> {
    Signal {
        signal_id: SignalId("u-history".into()),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(
                    0.,
                    SignalOrdinate::Real(SignalValue::Vector3([0.001, 0., 0.].map(Velocity::new))),
                ),
                SignalKnot::continuous(
                    1.,
                    SignalOrdinate::Real(SignalValue::Vector3([0.011, 0., 0.].map(Velocity::new))),
                ),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.),
            max_relative_error: Some(0.),
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: SignalProvenance::Authored,
    }
}
pub(super) fn accelerating_transient() -> (FluidTransientWorkload, SolverMeshArtifact) {
    let (mut w, m) = authored_transient();
    w.model.signals.push(PhysicalSignal::Velocity(ramp()));
    for constraint in &mut w.model.constraints {
        let AnalysisConstraint::Fluid(c) = constraint else {
            unreachable!()
        };
        let target = match &c.kind {
            FluidConstraintKind::NoSlipWall { target }
            | FluidConstraintKind::PrescribedVelocity { target, .. } => Some(target.clone()),
            _ => None,
        };
        if let Some(target) = target {
            c.kind = FluidConstraintKind::PrescribedVelocity {
                target,
                value: PrescribedValue::Signal(SignalId("u-history".into())),
            };
        }
    }
    w.model
        .initial_conditions
        .push(initial_velocity(InitialValue::SignalSample {
            signal_id: SignalId("u-history".into()),
            abscissa: 0.,
            approach: SignalSampleApproach::At,
        }));
    w.model
        .initial_conditions
        .push(InitialCondition::FluidPressure(InitialConditionValue {
            initial_condition_id: "original-p".into(),
            target: InitialConditionTarget::WholeModel,
            frame: CoordinateFrame::Global,
            value: InitialValue::Uniform(Pressure::new(101325.)),
            applies_before_step_id: "advance".into(),
        }));
    (w, m)
}
#[test]
fn transient_workload_retains_authored_initial_time_source_properties_and_cold_original_history() {
    let (w, m) = accelerating_transient();
    let cold_workload =
        FluidTransientWorkload::from_canonical_bytes(&w.canonical_bytes().unwrap()).unwrap();
    assert_eq!(cold_workload, w);
    let p = w.prepare(&m).unwrap();
    assert_eq!(p.initial_state().default_velocity_nodes(), 0);
    assert_eq!(
        p.initial_state().applied_condition_ids(),
        &["original-u", "original-p"]
    );
    assert_eq!(
        p.formulation().realization().temporals,
        vec![TemporalImplementation::FluidBackwardEulerConservativeMomentumPressure]
    );
    let budget = p.result_encoding_profile().unwrap();
    let result = p.execute().unwrap();
    let bytes = result.canonical_bytes().unwrap();
    assert!((bytes.len() as u64) <= budget.encoded_bytes_upper_bound);
    let projection = crate::topology::AnalysisProjection::derive(&m).unwrap();
    let topology =
        crate::topology::FiniteVolumeTopology::derive(&projection, w.geometry_policy).unwrap();
    for (i, r) in result.retained_results() {
        let time = 0.3 * ((i + 1) as f64 / 3.);
        for u in r.velocity_m_per_s.canonical_values() {
            assert!((u[0] - (0.001 + 0.01 * time)).abs() < 1e-8);
            assert!(u[1].abs() < 1e-8 && u[2].abs() < 1e-8);
        }
        for (cell, p) in topology
            .cells
            .iter()
            .zip(r.pressure_fluctuation_pa.canonical_values())
        {
            assert!((p + 0.01 * (cell.centroid_m[0] - 1.)).abs() < 2e-7);
        }
    }
    drop(p);
    let cold = FluidTransientSolverResult::from_canonical_bytes(&bytes).unwrap();
    cold.recover(&cold_workload, &m).unwrap();
    let mut changed = serde_json::to_value(&cold).unwrap();
    changed["initial"]["velocity_m_per_s"][0][0] = serde_json::json!(0.5);
    assert!(
        serde_json::from_value::<FluidTransientSolverResult>(changed)
            .unwrap()
            .recover(&w, &m)
            .is_err()
    );
    let mut foreign = w.clone();
    foreign.model.initial_conditions[0] = initial_velocity(InitialValue::UniformVector3([
        Velocity::new(0.002),
        Velocity::new(0.),
        Velocity::new(0.),
    ]));
    assert!(cold.recover(&foreign, &m).is_err());
    let mut low = w.clone();
    low.policy.maximum_resident_bytes = Some(1);
    assert!(low.prepare(&m).is_err());
}

fn velocity_field(mesh: &SolverMeshArtifact, values: Vec<f64>) -> InitialFieldBinding {
    use crate::execution::{
        FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, FieldTopologyIdentity,
        SequenceIdentity, SolverResultIdentity, StepIdentity,
    };
    use crate::results::field::*;
    let sequence = SequenceIdentity::new(Digest::sha256("original-initial-producer")).unwrap();
    let step = StepIdentity::new(sequence, Digest::sha256("original-field-step")).unwrap();
    let result =
        SolverResultIdentity::from_step_result(step, Digest::sha256("original-velocity-data"))
            .unwrap();
    let request = FieldRecoveryRequest::new(
        step,
        FieldRecoverySource::SolverResult(result),
        FieldTopologyIdentity::VolumeMesh {
            mesh_digest: mesh.canonical_digest,
        },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let field = RecoveredField::new(
        RecoveredFieldDefinition::new(
            "original-velocity",
            ObservationQuantity::FluidVelocity,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap(),
        RecoveredFieldSample::Final,
        AnalysisField {
            field_id: "original-velocity".into(),
            layout: runmat_analysis_core::AnalysisFieldLayout::vector3(
                runmat_analysis_core::ObservationQuantity::FluidVelocity,
                runmat_analysis_core::FieldLocation::Node,
            ),
            shape: vec![mesh.topology.nodes.len(), 3],
            values: AnalysisFieldValues::HostF64(values),
        },
    )
    .unwrap();
    let topology = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let artifact = RecoveredFieldArtifact::new(
        request.identity(),
        FieldRecoverySource::SolverResult(result),
        topology,
        RecoveredFieldEntityCounts {
            nodes: mesh.topology.nodes.len(),
            volume_elements: mesh.topology.volume_elements.len(),
            volume_integration_points: 0,
            oriented_edges: 0,
            oriented_faces: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap();
    let reference = InitialFieldArtifactRef {
        artifact_id: "published-original-velocity".into(),
        content_digest: artifact.identity().digest().to_string(),
        topology_digest: topology.to_string(),
        field_id: "original-velocity".into(),
    };
    InitialFieldBinding::new(reference, artifact).unwrap()
}
#[test]
fn nonuniform_original_nodal_field_integrates_exact_cell_and_face_initial_data_without_repair() {
    let (mut w, m) = authored_transient();
    let omega = 0.01;
    let field = velocity_field(
        &m,
        m.topology
            .nodes
            .iter()
            .flat_map(|n| {
                [
                    -omega * n.coordinates_m.canonical_value()[1],
                    omega * n.coordinates_m.canonical_value()[0],
                    0.,
                ]
            })
            .collect(),
    );
    w.model
        .initial_conditions
        .push(initial_velocity(InitialValue::FieldArtifact(
            field.reference().clone(),
        )));
    w.initial_fields.push(field);
    let p = w.prepare(&m).unwrap();
    let original = p.initial_state().original_fields();
    let projection = crate::topology::AnalysisProjection::derive(&m).unwrap();
    let t = crate::topology::FiniteVolumeTopology::derive(&projection, w.geometry_policy).unwrap();
    for (c, u) in t
        .cells
        .iter()
        .zip(original.velocity_m_per_s.canonical_values().iter())
    {
        assert!((u[0] + omega * c.centroid_m[1]).abs() < 1e-16);
        assert!((u[1] - omega * c.centroid_m[0]).abs() < 1e-16);
    }
    for (f, q) in t
        .faces
        .iter()
        .zip(original.face_volume_flux_m3_per_s.canonical_values().iter())
    {
        let v = [-omega * f.centroid_m[1], omega * f.centroid_m[0], 0.];
        let expected = (0..3)
            .map(|k| v[k] * f.owner_area_vector_m2[k])
            .sum::<f64>();
        assert!((q - expected).abs() < 1e-16);
    }
    let mut divergent = w.clone();
    let field = velocity_field(
        &m,
        m.topology
            .nodes
            .iter()
            .flat_map(|n| n.coordinates_m.canonical_value())
            .collect(),
    );
    divergent.model.initial_conditions[0] =
        initial_velocity(InitialValue::FieldArtifact(field.reference().clone()));
    divergent.initial_fields = vec![field];
    assert!(divergent.prepare(&m).is_err());
    let mut missing = w.clone();
    missing.initial_fields.clear();
    assert!(missing.prepare(&m).is_err());
    let mut conflict = w;
    let mut other = initial_velocity(InitialValue::UniformVector3([Velocity::new(1.); 3]));
    let InitialCondition::FluidVelocity(c) = &mut other else {
        unreachable!()
    };
    c.initial_condition_id = "conflicting-original-u".into();
    conflict.model.initial_conditions.push(other);
    assert!(conflict.prepare(&m).is_err());
}
#[test]
fn transient_formulation_cannot_be_claimed_by_generic_profile_admission() {
    let (w, m) = authored_transient();
    let identity = crate::admission::admit_canonical_fluid_transient(&m).unwrap();
    assert_ne!(
        identity.realization(),
        &crate::admission::canonical_fluid_steady_realization()
    );
    assert!(crate::admission::admit_nonstructural(
        PhysicsFamily::Fluid,
        AnalysisProcedure::TransientResponse,
        Some(&m),
        SourceImplementation::CanonicalVolumeProjection,
        BoundaryTransferImplementation::CanonicalBoundaryProjection
    )
    .is_err());
    assert_eq!(
        crate::admission::registered_formulations(
            PhysicsFamily::Fluid,
            AnalysisProcedure::TransientResponse
        )
        .len(),
        1
    );
    let _ = w;
}

/// The authored study workload runs the same backward-Euler history as the kernel workload,
/// retains it with its initial state, and cold-recovers every interval without solving.
#[test]
fn authored_study_workload_retains_and_recovers_the_complete_transient_history() {
    let (kernel, m) = accelerating_transient();
    let study = FluidSteadyWorkload {
        model: kernel.model.clone(),
        step_id: kernel.step_id.clone(),
        policy: kernel.policy,
        geometry_policy: kernel.geometry_policy,
    };
    assert!(study.is_transient());
    assert_eq!(
        study.formulation(&m).unwrap().realization().temporals,
        vec![TemporalImplementation::FluidBackwardEulerConservativeMomentumPressure]
    );
    let budget = study.result_encoding_profile(&m).unwrap();
    let result = study.execute(&m).unwrap();
    let bytes = result.canonical_bytes().unwrap();
    assert!((bytes.len() as u64) <= budget.encoded_bytes_upper_bound);
    let history = result.transient_history().unwrap();
    assert_eq!(history.intervals().len(), 3);
    let reference = kernel.execute(&m).unwrap();
    assert_eq!(history.intervals(), reference.intervals());
    assert_eq!(history.retained_results(), reference.retained_results());
    assert_eq!(result.original(), reference.terminal().unwrap());
    let cold = FluidSteadySolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &study,
        &m,
    )
    .unwrap();
    assert_eq!(cold, result);
    let receipt = FluidNumericalReceiptArtifact::new(&study, &cold, &m).unwrap();
    receipt.validate_against(&study, &cold, &m).unwrap();
    // A steady study cannot adopt a retained transient history, nor the reverse.
    let mut steady = study.clone();
    steady.model.sequence = AnalysisSequence::from_steps(
        "physical-time",
        vec![ProcedureStep::fluid_static("advance")],
    );
    steady.model.initial_conditions.clear();
    assert!(FluidSteadySolverResult::from_canonical_bytes_for(
        &bytes,
        Digest::sha256(&bytes),
        &steady,
        &m
    )
    .is_err());
}

// witness: FEA-FLUID-TRANSIENT-001/history @
//   incompressible_laminar_transient/canonical_tetrahedra/numerical_kernel; coverage: partial;
//   boundary: an authored time history retains the sampled interval's fields (and drops the
//   others), and native field recovery publishes them at their physical times beside the
//   terminal state, against the accelerating-flow reference.
#[test]
fn authored_history_retains_and_publishes_fields_at_sampled_times() {
    let (kernel, m) = accelerating_transient();
    let mut study = FluidSteadyWorkload {
        model: kernel.model.clone(),
        step_id: kernel.step_id.clone(),
        policy: kernel.policy,
        geometry_policy: kernel.geometry_policy,
    };
    study.model.sequence.steps[0]
        .observations
        .push(ObservationRequest::History(ObservationSpec {
            observation_id: ObservationId("velocity_history".into()),
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
            sampling: SamplingPolicy::TimeSeconds(vec![0.1]),
            reduction: None,
            storage: ObservationStoragePolicy {
                precision: SignalPrecision::Real64,
                compression: CompressionPolicy::Lossless,
                retention: RetentionPolicy::Canonical,
                partial_access: PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: ObservationPurpose::Publication,
            cost: ObservationCostPolicy::Automatic,
        }));
    let result = study.execute(&m).unwrap();
    let history = result.transient_history().unwrap();
    assert_eq!(history.intervals().len(), 3);
    assert_eq!(
        history
            .retained_results()
            .iter()
            .map(|(ordinal, _)| *ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let request = result.native_field_request(&study, &m).unwrap();
    let fields = result.recover_native_fields(&study, &m, &request).unwrap();
    result.validate_native_fields(&study, &m, &fields).unwrap();
    let samples = fields
        .fields()
        .iter()
        .map(|field| field.sample())
        .collect::<Vec<_>>();
    let velocity = fields
        .fields()
        .iter()
        .filter(|field| field.definition().quantity() == ObservationQuantity::FluidVelocity)
        .collect::<Vec<_>>();
    assert_eq!(velocity.len(), 2);
    for (field, time) in velocity.iter().zip([0.1, 0.3]) {
        let crate::results::field::RecoveredFieldSample::TimeSeconds(t) = field.sample() else {
            panic!("{:?}", field.sample())
        };
        assert!((t - time).abs() < 1e-12, "{t}");
        let values = field.storage().as_host_f64().unwrap();
        for u in values.chunks(3) {
            assert!(
                (u[0] - (0.001 + 0.01 * time)).abs() < 1e-8,
                "{u:?} at {time}"
            );
        }
    }
    assert!(samples.windows(2).all(|pair| match pair {
        [crate::results::field::RecoveredFieldSample::TimeSeconds(a),
         crate::results::field::RecoveredFieldSample::TimeSeconds(b)] => a <= b,
        _ => false,
    }));
}
