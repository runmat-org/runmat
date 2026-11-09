use super::*;

fn observation_fixture(
    order: runmat_meshing_core::ElementOrder,
    frequency: f64,
) -> (
    FrequencySamplePublicationAuthority,
    FrequencySamplePublicationSample,
    crate::results::observation::ObservationArtifact,
) {
    use crate::results::field::RecoveredFieldCatalog;
    use crate::results::observation::{
        evaluate_observation_sample, resolve_observation_plan, ObservationArtifact,
        ObservationArtifactInput, ObservationFieldSelection, ObservationLimits,
    };
    use runmat_analysis_core::*;
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let mut authority = authority(true, &[frequency]);
    authority.topology = FieldTopologyIdentity::Structural {
        topology: CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: projection.projection_digest,
        },
    };
    let mut sample = samples(&authority).remove(0);
    let mut recovery_plan = sample.recovery_request.plan.clone();
    recovery_plan.requirements[0].cost.source_entity_count = projection.nodes.len() as u64;
    recovery_plan.requirements[0].cost.output_entity_count = projection.nodes.len() as u64;
    sample.recovery_request = FieldRecoveryRequest::new(
        authority.producer,
        sample.fields.source,
        authority.topology,
        recovery_plan,
    )
    .unwrap();
    let coordinate = RecoveredFieldSample::FrequencyHertz(frequency);
    let definition = sample.fields.fields[0].definition.clone();
    let field = RecoveredField::new(
        definition.clone(),
        coordinate,
        AnalysisField::host_f64(
            "pressure",
            definition.layout(),
            vec![projection.nodes.len(), 2],
            vec![1.0; projection.nodes.len() * 2],
        ),
    )
    .unwrap();
    let catalog = RecoveredFieldCatalog::new(&projection, vec![field.clone()]).unwrap();
    sample.fields = RecoveredFieldArtifact::new(
        sample.recovery_request.identity(),
        sample.fields.source,
        projection.projection_digest,
        RecoveredFieldEntityCounts {
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: projection.elements.len() * 4,
            oriented_edges: 0,
            oriented_faces: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field],
    )
    .unwrap()
    .manifest()
    .unwrap();
    let mut step = ProcedureStep::acoustic_harmonic("observation-contract");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![frequency]);
    let observation_id = ObservationId("pressure.sample".into());
    step.observations = vec![ObservationRequest::History(ObservationSpec {
        observation_id: observation_id.clone(),
        quantity: ObservationQuantity::AcousticPressure,
        value_shape: ObservationValueShape::Scalar,
        target: ObservationTarget::WholeModel,
        source_location: FieldLocation::Node,
        output_location: FieldLocation::Node,
        recovery: RecoveryPolicy::Native,
        component: ObservationComponent::All,
        frame: CoordinateFrame::Global,
        sign_convention: SignConvention::Native,
        complex_representation: ComplexRepresentation::Complex,
        sampling: SamplingPolicy::FrequencyHertz(vec![frequency]),
        reduction: None,
        storage: ObservationStoragePolicy {
            precision: SignalPrecision::Complex128,
            compression: CompressionPolicy::Lossless,
            retention: RetentionPolicy::Canonical,
            partial_access: PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: ObservationPurpose::Publication,
        cost: ObservationCostPolicy::UpperBound(ObservationCostEstimate {
            operation_count: 100_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        }),
    })];
    let plan = resolve_observation_plan(
        &step,
        &projection,
        &[definition],
        ObservationLimits::default(),
    )
    .unwrap();
    let result =
        evaluate_observation_sample(&plan, &observation_id, &projection, &catalog, coordinate)
            .unwrap();
    let selection = ObservationFieldSelection::structural(
        &sample.recovery_request,
        &sample.fields,
        vec![sample.fields.fields[0].field],
    )
    .unwrap();
    let request =
        crate::execution::ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: request.identity(),
        producer: request.producer,
        source_selection: request.selection.identity(),
        source: request.selection.source(),
        topology_digest: plan.topology_digest,
        plan_identity: plan.identity,
        results: vec![result],
    })
    .unwrap();
    sample.observations = Some(artifact.manifest().unwrap());
    (authority, sample, artifact)
}

#[test]
fn optional_observation_catalog_is_bound_to_exact_recovered_sample() {
    use crate::results::observation::{ObservationArtifact, ObservationArtifactInput};
    use runmat_meshing_core::ElementOrder;
    let (authority, sample, artifact) = observation_fixture(ElementOrder::Tet4, 1.0);
    let publication =
        FrequencySamplePublicationManifest::new(authority.clone(), vec![sample.clone()]).unwrap();
    let cold = FrequencySamplePublicationManifest::from_canonical_bytes(
        &publication.canonical_bytes().unwrap(),
    )
    .unwrap();
    cold.validate_for(&authority).unwrap();
    assert_eq!(cold, publication);
    assert_eq!(cold.samples()[0].observations, sample.observations);
    for mutation in 0..6 {
        let mut source = artifact.source;
        let ObservationFieldSource::StructuralRecovery {
            recovery_request,
            artifact: field_artifact,
            numerical_source,
            topology: _,
        } = &mut source
        else {
            panic!("expected shared recovery source")
        };
        match mutation {
            0 => {
                *recovery_request = FieldRecoveryRequest::new(
                    producer("foreign recovery"),
                    sample.fields.source,
                    authority.topology,
                    sample.recovery_request.plan.clone(),
                )
                .unwrap()
                .identity()
            }
            1 => *field_artifact = samples(&self::authority(true, &[1.0]))[0].fields.identity,
            2 => {
                *numerical_source = FieldRecoverySource::SolverResult(
                    SolverResultIdentity::from_step_result(
                        authority.producer,
                        Digest::sha256("foreign observation source"),
                    )
                    .unwrap(),
                )
            }
            3 => {
                let (_, foreign_sample, _) = observation_fixture(ElementOrder::Tet10, 1.0);
                let mut substituted = sample.clone();
                substituted.observations = foreign_sample.observations;
                assert!(FrequencySamplePublicationManifest::new(
                    authority.clone(),
                    vec![substituted]
                )
                .is_err());
                continue;
            }
            4 | 5 => {}
            _ => unreachable!(),
        }
        // Rebuild through the shared constructor so each substituted observation manifest has
        // its own convergent identity. Rejection must come from the publication lineage check.
        let alternate = if mutation == 5 {
            observation_fixture(ElementOrder::Tet4, 2.0).2
        } else {
            artifact.clone()
        };
        if mutation == 5 {
            source = artifact.source;
        }
        let observation = ObservationArtifact::new(ObservationArtifactInput {
            evaluation_request: alternate.evaluation_request,
            producer: if mutation == 4 {
                producer("foreign observation producer")
            } else {
                artifact.producer
            },
            source_selection: alternate.source_selection,
            source,
            topology_digest: alternate.topology_digest,
            plan_identity: alternate.plan_identity,
            results: alternate.results().to_vec(),
        })
        .unwrap();
        observation.validate().unwrap();
        let mut substituted = sample.clone();
        substituted.observations = Some(observation.manifest().unwrap());
        assert!(
            FrequencySamplePublicationManifest::new(authority.clone(), vec![substituted]).is_err()
        );
    }
}
