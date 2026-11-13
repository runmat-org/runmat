//! Actual worker physics/authority evidence, not independent spatial qualification.
mod chain;
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_execution::{
    prepare_maxwell_harmonic_task, AnalysisOutputPolicy, MaxwellHarmonicTaskPreparation,
    PreparedMaxwellHarmonicTask,
};
use runmat_analysis_fea::{
    execution::MaxwellHarmonicSolverResult,
    fixtures::{fixture_model, FixtureId},
};
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::fixtures;

pub(crate) fn task(
    quadratic: bool,
    chunk: usize,
) -> (
    MaxwellHarmonicWorkload,
    SolverMeshArtifact,
    PreparedMaxwellHarmonicTask,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.cfd = None;
    let frequency = 1e8;
    model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
        reference_frequency: Frequency::new(frequency),
    });
    let mut step = ProcedureStep::electromagnetic_harmonic("field");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    step.state_outputs = vec![StateOutput {
        state_id: StateId("retained".into()),
        kind: StateKind::ElectromagneticComplex,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::Canonical,
    }];
    model.sequence = AnalysisSequence::from_steps("native-maxwell", vec![step]);
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(4.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: Vec::new(),
    });
    for boundary in ["slab_left", "slab_right", "slab_insulated"] {
        model
            .constraints
            .push(AnalysisConstraint::Electrical(ElectricalConstraint {
                constraint_id: boundary.into(),
                kind: ElectricalConstraintKind::ElectricPerfectConductor {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(boundary),
                    ),
                },
            }));
    }
    let vector = |value| SignalValue::Vector3([0.0, value, 0.0].map(ElectricCurrentDensity::new));
    model
        .signals
        .push(PhysicalSignal::ElectricCurrentDensity(Signal {
            signal_id: SignalId("J".into()),
            shape: SignalShape::Vector3,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Tabulated(TabulatedSignal {
                domain: TabulatedSignalDomain::FrequencyHertz,
                knots: vec![SignalKnot::continuous(
                    frequency,
                    SignalOrdinate::Complex {
                        real: vector(0.6),
                        imaginary: vector(0.8),
                    },
                )],
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::HoldEndpoint,
            }),
            precision: SignalPrecision::Complex64,
            provenance: SignalProvenance::Authored,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
        }));
    model
        .excitations
        .push(ExcitationAction::MagneticVolumeCurrentDensity(
            AnalysisExcitation {
                excitation_id: ExcitationId("current".into()),
                target: ExcitationTarget::WholeModel,
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::IntensityPerVolume,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("J".into()),
                active_step_ids: vec!["field".into()],
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: false,
                    preserve_moment: false,
                    relative_tolerance: 1e-12,
                },
            },
        ));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("current".into()),
        action_ids: vec![ExcitationId("current".into())],
        activation: ExcitationCaseActivation::Always,
    });
    let w = MaxwellHarmonicWorkload {
        model,
        step_id: ProcedureStepId("field".into()),
        relative_residual_target: 1e-10,
        frequency: runmat_analysis_core::Frequency::new(1e8),
    };
    let mesh = if quadratic {
        fixtures::quadratic_slab_mesh(2)
    } else {
        fixtures::slab_mesh(2)
    };
    let digest = Digest::sha256(b"native Maxwell complex fixture");
    let prepared = prepare_maxwell_harmonic_task(MaxwellHarmonicTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: digest,
        frozen_project_digest: None,
        program_revision: ProgramRevision::new(
            digest,
            digest,
            ProgramEnvironment::new(1, 1, digest, digest, "matlab").unwrap(),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "native-maxwell".into(),
            encryption_context: digest,
        },
        workload: w.clone(),
        mesh: mesh.clone(),
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap();
    (w, mesh, prepared)
}

fn invoke(task: &PreparedMaxwellHarmonicTask) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
    execute(AnalysisKernelInvocation {
        kind: AnalysisWorkKind::SolveMaxwellHarmonic,
        host: &task.host,
        inputs: &task.inputs,
        output_policy: task.output_policy,
    })
}

#[test]
fn native_both_orders_preserve_original_complex_fields_power_and_receipt_across_chunking() {
    for quadratic in [false, true] {
        let mut identity = None;
        for chunk in [1024, 8192] {
            let (w, mesh, task) = task(quadratic, chunk);
            let mut outputs = invoke(&task).unwrap().into_iter();
            let object = outputs
                .next()
                .unwrap()
                .into_prepared(
                    task.output_policy.chunk_bytes(),
                    task.output_policy.inventory_limits(),
                )
                .unwrap();
            let bytes = object.payload();
            let result = MaxwellHarmonicSolverResult::from_canonical_bytes_for(
                &bytes,
                Digest::sha256(&bytes),
                &w,
                &mesh,
            )
            .unwrap();
            if let Some(previous) = identity {
                assert_eq!(previous, result.identity().unwrap());
            }
            identity = Some(result.identity().unwrap());
            let response = result.response();
            assert!(response.ohmic_power_w > runmat_types::physical::Quantity::new(0.0));
            assert!(
                response.electric_energy_j > runmat_types::physical::Quantity::new(0.0)
                    && response.magnetic_energy_j > runmat_types::physical::Quantity::new(0.0)
            );
            assert!(response.relative_physical_residual < 8e-10);
            assert!(response.relative_complex_power_error < 1.6e-9);
            assert!(
                (response.source_power_va.canonical_value().real
                    - response.ohmic_power_w.canonical_value())
                .abs()
                    <= 2e-9 * response.ohmic_power_w.canonical_value()
            );
            assert!(response
                .samples
                .iter()
                .flat_map(|s| s.electric_field_v_per_m.canonical_value())
                .any(|v| v.real != 0.0 && v.imaginary != 0.0));
            let request = result.native_field_request(&w, &mesh).unwrap();
            let fields = result.recover_native_fields_from_result(&request).unwrap();
            // Edge circulation, five integration-point fields, their loss, and four element averages.
            assert_eq!(fields.fields().len(), 11);
            assert_eq!(
                fields.entity_counts.volume_integration_points,
                mesh.topology.volume_elements.len() * 15
            );
            let expected = response
                .samples
                .iter()
                .flat_map(|s| s.electric_field_v_per_m.canonical_value())
                .flat_map(|v| [v.real, v.imaginary])
                .collect::<Vec<_>>();
            assert_eq!(
                fields.fields()[1].storage().as_host_f64().unwrap(),
                expected
            );
            let receipt = outputs.next().unwrap();
            MaxwellNumericalReceiptArtifact::from_canonical_bytes(receipt.whole_payload().unwrap())
                .unwrap()
                .validate_against(&w, &result, &mesh)
                .unwrap();
            assert!(outputs.next().is_some());
            assert!(outputs.next().is_none());
        }
    }
}

#[test]
fn native_rejects_source_closure_output_policy_and_understated_plan_before_solve() {
    let (mut w, _, task) = task(false, 1024);
    let invoke_inputs = |inputs: &[runmat_analysis_execution::PreparedAnalysisObjects], policy| {
        execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveMaxwellHarmonic,
            host: &task.host,
            inputs,
            output_policy: policy,
        })
    };
    assert!(invoke_inputs(&task.inputs[..2], task.output_policy).is_err());
    // Leave the immutable host and manifests intact but corrupt an original chunk.
    // Admission must reject its digest before attempting semantic source decoding.
    let mut corrupted = task.inputs.clone();
    corrupted[0].objects[1].bytes[0] ^= 1;
    assert!(matches!(
        invoke_inputs(&corrupted, task.output_policy),
        Err(AnalysisBridgeError::Identity(_)) | Err(AnalysisBridgeError::Artifact(_))
    ));
    assert!(invoke_inputs(
        &task.inputs,
        AnalysisOutputPolicy::new(8192, task.output_policy.inventory_limits()).unwrap()
    )
    .is_err());
    w.relative_residual_target *= 0.1;
    let mut inputs = task.inputs.clone();
    inputs[0] = prepare_analysis_objects(
        AnalysisNumericalObjectKind::MaxwellHarmonicWorkload,
        w.identity().unwrap().digest(),
        &w.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::InputValue,
        task.output_policy.inventory_limits(),
    )
    .unwrap();
    assert!(invoke_inputs(&inputs, task.output_policy).is_err());
    let mut plan = task.resource_plan.clone();
    plan.request.memory_bytes = 1;
    inputs = task.inputs.clone();
    let bytes = runmat_canonical_codec::encode_contract(
        b"runmat-fea-resource-plan/v1\0",
        "resource-plan-artifact",
        &plan,
        runmat_canonical_codec::CanonicalLimits::new(64 * 1024, 8192, 65536, 64),
    )
    .unwrap();
    inputs[2] = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        plan.identity.digest(),
        &bytes,
        1024,
        ObjectNamespace::InputValue,
        task.output_policy.inventory_limits(),
    )
    .unwrap();
    assert!(invoke_inputs(&inputs, task.output_policy).is_err());
    // Rehash the outer transport closure coherently: the internal resource authority must
    // still reject understatement rather than relying only on the original manifest mismatch.
    let old = &task.host.workload;
    let request = runmat_analysis_execution::AnalysisWorkloadRequest::new(
        old.study,
        old.step,
        old.execution_plan,
        old.kind,
        inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: ValueRefKind::DriverObject,
            })
            .collect(),
        old.outputs.clone(),
        old.required_capabilities.clone(),
    )
    .unwrap();
    let host = runmat_analysis_execution::AnalysisHostWorkload::new(
        request,
        task.host.artifact_access.clone(),
        task.host.output_policy,
    )
    .unwrap();
    assert!(execute(AnalysisKernelInvocation {
        kind: old.kind,
        host: &host,
        inputs: &inputs,
        output_policy: task.output_policy
    })
    .is_err());
}
