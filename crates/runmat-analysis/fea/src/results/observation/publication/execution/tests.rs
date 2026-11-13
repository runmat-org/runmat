use super::*;
use crate::execution::{
    FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoverySource, SequenceIdentity,
    SolverResultIdentity,
};
use crate::results::field::{
    canonical_field_partitions, RecoveredFieldArtifact, RecoveredFieldEntityCounts,
    RecoveredFieldPartitionSet,
};
use crate::results::observation::partition_tests;
use runmat_analysis_core::{ObservationRequest, ObservationTarget};

fn fixture() -> ObservationPublicationCatalog {
    let spec = partition_tests::spec("execution", ObservationTarget::WholeModel);
    let (projection, plan, field) =
        partition_tests::prepared(ObservationRequest::Field(spec.clone()));
    let producer = StepIdentity::new(
        SequenceIdentity::new(Digest::sha256("execution sequence")).unwrap(),
        Digest::sha256("execution step"),
    )
    .unwrap();
    let solver =
        SolverResultIdentity::from_step_result(producer, Digest::sha256("execution solver"))
            .unwrap();
    let recovery = FieldRecoveryRequest::new(
        producer,
        FieldRecoverySource::SolverResult(solver),
        crate::execution::FieldTopologyIdentity::Structural { topology: crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Explicit {
            structural_topology_digest: plan.topology_digest,
        } },
        FieldRecoveryPlan::default(),
    )
    .unwrap();
    let manifest = RecoveredFieldArtifact::new(
        recovery.identity(),
        recovery.source,
        plan.topology_digest,
        RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        },
        vec![field.clone()],
    )
    .unwrap()
    .manifest()
    .unwrap();
    let source = ObservationFieldCatalog::structural(recovery, manifest).unwrap();
    let selection = source
        .canonical_selection(&plan, ObservationSelectionCoordinate::StructuralRecovery)
        .unwrap();
    let request = ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
    let policy = ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(2).unwrap());
    let partitions = ObservationPartitionPlan::derive(&plan, &spec.observation_id, policy).unwrap();
    let fields =
        RecoveredFieldPartitionSet::new(canonical_field_partitions(&field).unwrap()).unwrap();
    let contributions = partitions
        .partitions
        .iter()
        .map(|partition| {
            evaluate_solid_observation_partition(
                &plan,
                &partitions,
                partition,
                &projection,
                &fields,
                None,
            )
            .unwrap()
        })
        .collect();
    let result = reduce_observation_partitions(&plan, &partitions, contributions).unwrap();
    let groups = partitions
        .partitions
        .iter()
        .map(|partition| {
            let indices = solid_observation_partition_source_indices(
                &plan,
                &partitions,
                partition,
                &projection,
            )
            .unwrap();
            select_canonical_field_partitions(&request.selection.fields()[0], &indices)
                .unwrap()
                .descriptors
                .iter()
                .map(|item| item.partition)
                .collect()
        })
        .collect();
    let record = ObservationResultExecution {
        result: result.logical_identity,
        source_partitions: groups,
    };
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: request.identity(),
        producer,
        source_selection: request.selection.identity(),
        source: request.selection.source(),
        topology_digest: plan.topology_digest,
        plan_identity: plan.identity,
        results: vec![result],
    })
    .unwrap();
    let sample = ObservationSamplePublication::new(request, &artifact).unwrap();
    let mut catalog = ObservationPublicationCatalog {
        schema_version: 2,
        identity: ObservationPublicationCatalogIdentity(Digest::sha256([])),
        scope: ObservationPublicationScope::Complete,
        producer,
        plan,
        source,
        declarations: vec![ObservationRequest::Field(spec)],
        entries: vec![ObservationPublicationEntry::Sample(Box::new(sample))],
        execution_policy: policy,
        execution: vec![record],
    };
    catalog.identity = catalog.derive_identity().unwrap();
    catalog.validate().unwrap();
    catalog
}

#[test]
fn execution_visitor_reconstructs_canonical_partition_sources() {
    let catalog = fixture();
    let mut visits = 0;
    catalog
        .visit_execution(|source, plan, evidence, sets| {
            visits += 1;
            assert_eq!(sets.len(), plan.partitions.len());
            assert_eq!(sets.len(), evidence.contributions.len());
            for (partition, set) in plan.partitions.iter().zip(sets) {
                crate::execution::ObservationPartitionEvaluationRequest::published_identity(
                    source,
                    plan,
                    partition,
                    set.identity,
                    evidence.event_attribution,
                )?;
                assert!(!set.descriptors.is_empty());
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(visits, 1);
}

#[test]
fn execution_metadata_rejects_rehashed_coverage_policy_and_group_changes() {
    let original = fixture();
    let reject = |mut catalog: ObservationPublicationCatalog| {
        catalog.identity = catalog.derive_identity().unwrap();
        assert!(catalog.validate().is_err());
    };
    let mut missing = original.clone();
    missing.execution.clear();
    reject(missing);
    let mut extra = original.clone();
    extra.execution.push(extra.execution[0].clone());
    reject(extra);
    let mut policy = original.clone();
    policy.execution_policy = ObservationExecutionPolicy::serial();
    reject(policy);
    let mut group = original.clone();
    group.execution[0].source_partitions.pop();
    reject(group);
    let mut duplicate = original;
    let block = duplicate.execution[0].source_partitions[0][0];
    duplicate.execution[0].source_partitions[0].push(block);
    reject(duplicate);
}
