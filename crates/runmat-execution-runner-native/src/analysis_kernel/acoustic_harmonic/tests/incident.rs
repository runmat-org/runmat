//! Native incident-pressure execution and detached cold publication, not continuum evidence.
use super::*;
use runmat_analysis_core::{
    AnalysisExcitation, AnalysisFieldValues, CoordinateFrame, ExcitationAction, ExcitationCase,
    ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution, ExcitationId,
    ExcitationKinematics, ExcitationTarget, PhysicalSignal, ProjectionRequirement, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalId, SignalInterpolation, SignalKnot,
    SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape,
    SignalValue, TabulatedSignal, TabulatedSignalDomain,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_acoustic_harmonic_result, import_semantic_state_completion,
    prepare_semantic_state_publication, AnalysisHostResponse, SemanticStatePublicationPreparation,
};
use runmat_analysis_fea::execution::{CanonicalSemanticState, SemanticStatePublicationSource};
use runmat_execution_artifact::cache::FilesystemObjectStore;

fn incident_task(order: ElementOrder, chunk: usize) -> PreparedAcousticHarmonicTask {
    let (mut workload, mesh, original) = task(order, chunk);
    workload.model.constraints.clear();
    workload
        .model
        .signals
        .push(PhysicalSignal::Pressure(Signal {
            signal_id: SignalId("incident-drive".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Tabulated(TabulatedSignal {
                domain: TabulatedSignalDomain::FrequencyHertz,
                knots: [0.01, 1.0]
                    .into_iter()
                    .map(|frequency| {
                        SignalKnot::continuous(
                            frequency,
                            SignalOrdinate::Complex {
                                real: SignalValue::Scalar(Pressure::new(1.0)),
                                imaginary: SignalValue::Scalar(Pressure::new(0.4)),
                            },
                        )
                    })
                    .collect(),
                interpolation: SignalInterpolation::Linear,
                extrapolation: SignalExtrapolation::Reject,
            }),
            precision: SignalPrecision::Complex64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 2,
            },
            provenance: SignalProvenance::Authored,
        }));
    workload
        .model
        .excitations
        .push(ExcitationAction::AcousticIncidentPressure(
            AnalysisExcitation {
                excitation_id: ExcitationId("incident-inlet".into()),
                target: ExcitationTarget::Region(RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("face:0"),
                )),
                frame: CoordinateFrame::BoundaryNormalTangent,
                direction: None,
                distribution: ExcitationDistribution::UniformKinematic,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("incident-drive".into()),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1e-10,
                },
            },
        ));
    workload.model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("incident-inlet".into()),
        action_ids: vec![ExcitationId("incident-inlet".into())],
        activation: ExcitationCaseActivation::Always,
    });
    // Rebuild authenticated input authority after changing the physical source. The existing
    // face:3 impedance remains rho*c=6; inlet admittance belongs to canonical source lowering.
    prepare_acoustic_harmonic_task(AcousticHarmonicTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: Digest::sha256("acoustic-native-incident-test"),
        frozen_project_digest: None,
        program_revision: original.program.request.recipe.program_revision,
        artifact_access: original.host.artifact_access,
        workload,
        mesh,
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap()
}

#[test]
fn native_incident_pressure_source_and_cold_publication_are_transport_independent() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut expected_result = None;
        let mut expected_state = None;
        for chunk in [1024, 8192] {
            let task = incident_task(order, chunk);
            let limits = ObjectInventoryLimits::default();
            let directory = tempfile::tempdir().unwrap();
            let object_path = directory.path().join("objects");
            let mut store =
                FilesystemObjectStore::open(&object_path, limits.max_object_bytes).unwrap();
            task.program.write_to(&mut store).unwrap();
            let completed = execute_serial_analysis(
                &task.program.request,
                &mut store,
                &NativeAnalysisKernel,
                chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&task.host, &completed)
                .unwrap()
                .program_response();
            drop(store);
            let mut store =
                FilesystemObjectStore::open(&object_path, limits.max_object_bytes).unwrap();
            let imported =
                import_acoustic_harmonic_result(&store, response, &task, limits).unwrap();
            let result = &imported.result;
            let identity = result.source_result_identity().unwrap();
            assert_eq!(
                identity.digest(),
                imported.objects.manifest.logical_identity
            );
            if let Some(expected) = expected_result {
                assert_eq!(identity, expected);
            } else {
                expected_result = Some(identity);
            }
            assert!(result
                .pressure_pa()
                .iter()
                .any(|p| p.real.abs() + p.imaginary.abs() > 1e-10));
            assert!(result.velocity_m_per_s().iter().any(|v| v
                .real
                .iter()
                .chain(&v.imaginary)
                .any(|x| x.abs() > 1e-10)));
            let requests = result.state_publication_requests().unwrap();
            assert_eq!(requests.len(), 1);
            let request = &requests[0];
            let publication =
                prepare_semantic_state_publication(SemanticStatePublicationPreparation {
                    source_host: task.host.clone(),
                    source_result: imported.objects.clone(),
                    source_profiles: imported.source_profiles,
                    source: SemanticStatePublicationSource::AcousticNativeFields,
                    declaration: request.declaration.clone(),
                    topology_digest: request.topology_digest,
                    program_revision: task.program.request.recipe.program_revision.clone(),
                    chunk_bytes: chunk,
                    inventory_limits: limits,
                })
                .unwrap();
            assert_eq!(
                publication.host.workload.kind,
                AnalysisWorkKind::PublishState
            );
            assert!(publication.inputs.iter().all(|input| !matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::Mesh
                    | AnalysisNumericalObjectKind::AcousticHarmonicWorkload
            )));
            publication.program.write_to(&mut store).unwrap();
            let completed = execute_serial_analysis(
                &publication.program.request,
                &mut store,
                &NativeAnalysisKernel,
                chunk,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&publication.host, &completed)
                .unwrap()
                .program_response();
            drop(store);
            let store = FilesystemObjectStore::open(&object_path, limits.max_object_bytes).unwrap();
            let published =
                import_semantic_state_completion(&store, response, &publication, limits).unwrap();
            let state =
                CanonicalSemanticState::from_canonical_bytes(&published.objects.payload()).unwrap();
            assert_eq!(state, published.state);
            assert_eq!(state.source_result, identity);
            assert_eq!(state.producer, task.host.workload.step);
            if let Some(expected) = expected_state {
                assert_eq!(state.identity, expected);
            } else {
                expected_state = Some(state.identity);
            }
            let payload = state.payload.acoustic().unwrap();
            assert_eq!(payload.ordered_node_ids, result.ordered_node_ids());
            assert_eq!(payload.ordered_element_ids, result.ordered_element_ids());
            let fields = payload.fields.fields();
            // Pressure, particle velocity, active intensity and plane-wave port intensity.
            assert_eq!(fields.len(), 4);
            let original_pressure = result
                .pressure_pa()
                .iter()
                .flat_map(|p| [p.real, p.imaginary])
                .collect::<Vec<_>>();
            let original_velocity = result
                .velocity_m_per_s()
                .iter()
                .flat_map(|v| (0..3).flat_map(move |axis| [v.real[axis], v.imaginary[axis]]))
                .collect::<Vec<_>>();
            for (field, original) in fields.iter().zip([original_pressure, original_velocity]) {
                let AnalysisFieldValues::HostF64(values) = &field.storage().values else {
                    panic!("native complex acoustic publication must retain host-backed fields");
                };
                assert_eq!(
                    values, &original,
                    "cold publication preserves original complex pressure/velocity"
                );
            }
        }
    }
}
