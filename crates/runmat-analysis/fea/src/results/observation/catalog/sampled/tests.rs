//! Metadata/publication lineage witnesses; not independent magnetic discretization evidence.
use super::*;
use crate::{
    execution::{
        AccuracySourceKind, MagnetostaticWorkload, ObservationEvaluationRequest,
        ObservationHistorySourceContext,
    },
    physics::magnetic::test_support::authored_model,
    results::{
        field::{
            RecoveredField, RecoveredFieldArtifact, RecoveredFieldCatalog,
            RecoveredFieldDefinition, RecoveredFieldSample, VolumeFieldSamplingRule,
        },
        observation::{
            evaluate_sampled_observations, partition_tests::spec, resolve_sampled_observation_plan,
            ObservationArtifact, ObservationArtifactInput, ObservationLimits,
            ObservationPublicationCatalog, ObservationPublicationEntry,
            ObservationSamplePublication,
        },
    },
};
use runmat_analysis_core::{
    AnalysisField, ExtremaTiePolicy, FieldLocation, ObservationQuantity, ObservationRequest,
    ObservationTarget, ObservationValueShape, ProcedureStepId, ReductionPolicy, SeriesReduction,
    SpatialReduction,
};
use runmat_meshing_core::fixtures::{quadratic_slab_mesh, slab_mesh};

fn publish_spoof(
    workload: &MagnetostaticWorkload,
    mesh: &SolverMeshArtifact,
    request: FieldRecoveryRequest,
    fields: RecoveredFieldArtifact,
    sampling: SampledVolumeFieldTopology,
) -> AnalysisExecutionResult<ObservationPublicationCatalog> {
    let step = &workload.model.sequence.steps[0];
    let catalog = ObservationFieldCatalog::sampled_recovery(
        request,
        fields.manifest().unwrap(),
        sampling.clone(),
    )
    .unwrap();
    let definitions = fields
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let plan = resolve_sampled_observation_plan(
        step,
        &sampling,
        mesh,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    catalog
        .validate_sampled_plan(&plan, step, workload.producer_identity().unwrap(), mesh)
        .unwrap();
    let selection = catalog
        .canonical_selection(&plan, ObservationSelectionCoordinate::SampledRecovery)
        .unwrap();
    let retained = RecoveredFieldCatalog::new_for_topology(
        fields.topology_digest,
        fields.entity_counts,
        fields.into_fields(),
    )
    .unwrap();
    let results = evaluate_sampled_observations(&plan, &sampling, mesh, &retained).unwrap();
    let evaluation = ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
    let artifact = ObservationArtifact::new(ObservationArtifactInput {
        evaluation_request: evaluation.identity(),
        producer: evaluation.producer,
        source_selection: evaluation.selection.identity(),
        source: evaluation.selection.source(),
        topology_digest: plan.topology_digest,
        plan_identity: plan.identity,
        results,
    })
    .unwrap();
    let sample = ObservationSamplePublication::new(evaluation, &artifact).unwrap();
    let publication = ObservationPublicationCatalog::new(
        workload.producer_identity().unwrap(),
        plan,
        catalog,
        step,
        vec![ObservationPublicationEntry::Sample(Box::new(sample))],
        ObservationHistorySourceContext::Magnetostatic { workload, mesh },
    )?;
    let bytes = publication.metadata_validation_resident_bytes()?;
    let (verified, _, peak) =
        crate::topology::measure_test_allocations(|| publication.validate_metadata_admitted(bytes));
    verified?;
    assert!(
        peak as u64 <= bytes,
        "catalog metadata {peak} exceeds {bytes}"
    );
    let (short, retained, peak) = crate::topology::measure_test_allocations(|| {
        publication.validate_metadata_admitted(bytes - 1)
    });
    assert!(matches!(
        short,
        Err(AnalysisExecutionError::Limit(
            "catalog metadata validation workspace"
        ))
    ));
    assert_eq!(retained, 0);
    // Profiling custom digest serialization can allocate bounded identifier scratch.
    assert!((peak as u64) < bytes);
    Ok(publication)
}

fn fixture(
    mesh: &SolverMeshArtifact,
) -> (
    MagnetostaticWorkload,
    FieldRecoveryRequest,
    RecoveredFieldArtifact,
    ObservationFieldCatalog,
    ResolvedObservationPlan,
) {
    let mut workload = MagnetostaticWorkload {
        model: authored_model(-2.0),
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    };
    let mut flux = spec("flux", ObservationTarget::WholeModel);
    flux.quantity = ObservationQuantity::MagneticFluxDensity;
    flux.value_shape = ObservationValueShape::Vector3;
    flux.source_location = FieldLocation::IntegrationPoint;
    flux.output_location = FieldLocation::IntegrationPoint;
    let mut edge = spec("edge_max", ObservationTarget::WholeModel);
    edge.quantity = ObservationQuantity::MagneticEdgeCirculation;
    edge.source_location = FieldLocation::Edge;
    edge.output_location = FieldLocation::Edge;
    edge.reduction = Some(ReductionPolicy {
        spatial: SpatialReduction::Maximum,
        series: SeriesReduction::None,
        extrema_ties: ExtremaTiePolicy::AllLocations,
    });
    workload.model.sequence.steps[0].observations = vec![
        ObservationRequest::Field(flux),
        ObservationRequest::Reduction(edge),
    ];
    let result = workload.execute(mesh).unwrap();
    let recovery = result.native_field_request(&workload, mesh).unwrap();
    let fields = result.recover_native_fields_from_result(&recovery).unwrap();
    let sampling = result.sampling_topology().clone();
    let catalog = ObservationFieldCatalog::sampled_recovery(
        recovery.clone(),
        fields.manifest().unwrap(),
        sampling.clone(),
    )
    .unwrap();
    let definitions = fields
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let plan = resolve_sampled_observation_plan(
        &workload.model.sequence.steps[0],
        &sampling,
        mesh,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    (workload, recovery, fields, catalog, plan)
}

#[test]
fn sampled_original_retained_catalog_selection_publication_survive_cold_metadata_reads() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let (workload, recovery, fields, catalog, plan) = fixture(&mesh);
        let step = &workload.model.sequence.steps[0];
        let producer = workload.producer_identity().unwrap();
        let cold =
            ObservationFieldCatalog::from_canonical_bytes(&catalog.canonical_bytes().unwrap())
                .unwrap();
        assert_eq!(catalog, cold);
        cold.validate_sampled_plan(&plan, step, producer, &mesh)
            .unwrap();
        let selection = cold
            .canonical_selection(&plan, ObservationSelectionCoordinate::SampledRecovery)
            .unwrap();
        assert_eq!(selection.fields().len(), 2);
        cold.validate_canonical_selection(&plan, &selection)
            .unwrap();
        assert_eq!(
            selection.source().solver_result(),
            recovery.source.solver_result()
        );
        assert!(matches!(
            selection.source(),
            super::super::super::ObservationFieldSource::SampledRecovery { .. }
        ));
        assert!(cold.validate_structural_selection(&selection).is_err());
        let ObservationFieldCatalogSource::SampledRecovery { sampling, .. } = cold.source() else {
            panic!("sampled source")
        };
        let retained = RecoveredFieldCatalog::new_for_topology(
            fields.topology_digest,
            fields.entity_counts,
            fields.clone().into_fields(),
        )
        .unwrap();
        let results = evaluate_sampled_observations(&plan, sampling, &mesh, &retained).unwrap();
        let evaluation = ObservationEvaluationRequest::new(selection, plan.clone()).unwrap();
        let artifact = ObservationArtifact::new(ObservationArtifactInput {
            evaluation_request: evaluation.identity(),
            producer,
            source_selection: evaluation.selection.identity(),
            source: evaluation.selection.source(),
            topology_digest: plan.topology_digest,
            plan_identity: plan.identity,
            results,
        })
        .unwrap();
        let sample = ObservationSamplePublication::new(evaluation, &artifact).unwrap();
        let context = ObservationHistorySourceContext::Magnetostatic {
            workload: &workload,
            mesh: &mesh,
        };
        let publication = ObservationPublicationCatalog::new(
            producer,
            plan,
            cold,
            step,
            vec![ObservationPublicationEntry::Sample(Box::new(sample))],
            context,
        )
        .unwrap();
        let cold_publication = ObservationPublicationCatalog::from_canonical_bytes(
            &publication.canonical_bytes().unwrap(),
        )
        .unwrap();
        assert_eq!(publication, cold_publication);
        cold_publication.validate_for(step, context).unwrap();
        cold_publication
            .validate_source_result(
                producer,
                AccuracySourceKind::Magnetostatic,
                recovery.source.solver_result().unwrap(),
            )
            .unwrap();
        assert!(cold_publication
            .validate_source_result(
                producer,
                AccuracySourceKind::ElectricalStatic,
                recovery.source.solver_result().unwrap()
            )
            .is_err());
        let mut foreign_workload = workload.clone();
        foreign_workload.relative_residual_target = 1e-9;
        // Producer identity authenticates authored sequence/step, not all numerical intent.
        // Original workload-root and retained-result validation own tolerance/material binding.
        assert_eq!(foreign_workload.producer_identity().unwrap(), producer);
        assert_ne!(
            foreign_workload.identity().unwrap(),
            workload.identity().unwrap()
        );
        foreign_workload.model.sequence.sequence_id = "foreign-sequence".into();
        assert_ne!(foreign_workload.producer_identity().unwrap(), producer);
        assert!(cold_publication
            .validate_for(
                step,
                ObservationHistorySourceContext::Magnetostatic {
                    workload: &foreign_workload,
                    mesh: &mesh
                }
            )
            .is_err());
        assert!(cold_publication
            .validate_for(
                step,
                ObservationHistorySourceContext::Magnetostatic {
                    workload: &workload,
                    mesh: &slab_mesh(1)
                }
            )
            .is_err());
    }
}

#[test]
fn sampled_catalog_rejects_foreign_layout_manifest_inventory_and_original_step() {
    for mesh in [slab_mesh(2), quadratic_slab_mesh(2)] {
        let (workload, recovery, fields, catalog, plan) = fixture(&mesh);
        let ObservationFieldCatalogSource::SampledRecovery { sampling, .. } = catalog.source()
        else {
            panic!("sampled source")
        };
        let four = SampledVolumeFieldTopology::derive(
            &mesh,
            VolumeFieldSamplingRule::TetrahedronFourPoint,
        )
        .unwrap();
        assert!(ObservationFieldCatalog::sampled_recovery(
            recovery.clone(),
            fields.manifest().unwrap(),
            four
        )
        .is_err());
        let incomplete = RecoveredFieldArtifact::new(
            recovery.identity(),
            recovery.source,
            fields.topology_digest,
            fields.entity_counts,
            fields.fields()[1..].to_vec(),
        )
        .unwrap();
        assert!(ObservationFieldCatalog::sampled_recovery(
            recovery.clone(),
            incomplete.manifest().unwrap(),
            sampling.as_ref().clone()
        )
        .is_err());
        let mut altered_step = workload.model.sequence.steps[0].clone();
        altered_step.observations.pop();
        assert!(catalog
            .validate_sampled_plan(
                &plan,
                &altered_step,
                workload.producer_identity().unwrap(),
                &mesh
            )
            .is_err());
        let mut foreign = workload.clone();
        foreign.relative_residual_target = 1e-9;
        assert_eq!(
            foreign.producer_identity().unwrap(),
            workload.producer_identity().unwrap()
        );
        assert_ne!(foreign.identity().unwrap(), workload.identity().unwrap());
        foreign.model.sequence.sequence_id = "foreign-sequence".into();
        assert_ne!(
            foreign.producer_identity().unwrap(),
            workload.producer_identity().unwrap()
        );
        assert!(catalog
            .validate_sampled_plan(
                &plan,
                &workload.model.sequence.steps[0],
                foreign.producer_identity().unwrap(),
                &mesh
            )
            .is_err());
        // Fresh inner hashes prove cardinality rejection is relational, not stale integrity.
        let mut false_cost = recovery.plan.clone();
        false_cost.requirements[0].cost.source_entity_count += 1;
        false_cost.requirements[0].cost.output_entity_count += 1;
        false_cost.requirements[0].cost.arithmetic_operation_count += 1;
        let malformed = FieldRecoveryRequest::new(
            recovery.producer,
            recovery.source,
            recovery.topology,
            false_cost,
        )
        .unwrap();
        let false_manifest = RecoveredFieldArtifact::new(
            malformed.identity(),
            malformed.source,
            fields.topology_digest,
            fields.entity_counts,
            fields.fields().to_vec(),
        )
        .unwrap()
        .manifest()
        .unwrap();
        assert!(ObservationFieldCatalog::sampled_recovery(
            malformed,
            false_manifest,
            sampling.as_ref().clone()
        )
        .is_err());
        let all = ObservationFieldSelection::sampled_recovery(
            &recovery,
            &fields.manifest().unwrap(),
            sampling,
            fields
                .fields()
                .iter()
                .map(|field| field.identity())
                .collect(),
        )
        .unwrap();
        catalog.validate_sampled_selection(&all).unwrap();
        assert!(catalog.validate_canonical_selection(&plan, &all).is_err());
    }
}

#[test]
fn magnetic_publication_context_rejects_coherent_foreign_rule_and_native_field_id() {
    let mesh = slab_mesh(2);
    let (workload, request, fields, catalog, _) = fixture(&mesh);
    let ObservationFieldCatalogSource::SampledRecovery { sampling, .. } = catalog.source() else {
        panic!("sampled source")
    };
    let mut renamed = fields.fields().to_vec();
    let old = &renamed[2];
    let definition = RecoveredFieldDefinition::new(
        "foreign.flux",
        old.definition().quantity(),
        old.definition().location(),
        old.definition().entity_domain(),
        old.definition().value_shape().clone(),
        old.definition().numeric_representation(),
    )
    .unwrap();
    renamed[2] = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "foreign.flux",
            definition.layout(),
            old.storage().shape.clone(),
            old.storage().as_host_f64().unwrap().to_vec(),
        ),
    )
    .unwrap();
    let renamed = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        fields.topology_digest,
        fields.entity_counts,
        renamed,
    )
    .unwrap();
    assert!(publish_spoof(
        &workload,
        &mesh,
        request.clone(),
        renamed,
        sampling.as_ref().clone()
    )
    .is_err());

    // A valid generic four-point catalog must stay generic; the magnetic adapter rejects it.
    let four =
        SampledVolumeFieldTopology::derive(&mesh, VolumeFieldSamplingRule::TetrahedronFourPoint)
            .unwrap();
    let mut plan = request.plan.clone();
    let remapped = fields
        .fields()
        .iter()
        .zip(&mut plan.requirements)
        .map(|(field, requirement)| {
            let count = four.entity_counts().entity_count(requirement.entity_domain);
            let components = requirement.value_shape.component_count().unwrap();
            requirement.cost.source_entity_count = count as u64;
            requirement.cost.output_entity_count = count as u64;
            requirement.cost.arithmetic_operation_count = (count * components) as u64;
            let shape = if components == 1 {
                vec![count]
            } else {
                vec![count, components]
            };
            RecoveredField::new(
                field.definition().clone(),
                RecoveredFieldSample::Final,
                AnalysisField::host_f64(
                    field.definition().field_id(),
                    field.definition().layout(),
                    shape,
                    vec![0.0; count * components],
                ),
            )
            .unwrap()
        })
        .collect();
    let request = FieldRecoveryRequest::new(
        request.producer,
        request.source,
        four.identity().unwrap(),
        plan,
    )
    .unwrap();
    let remapped = RecoveredFieldArtifact::new(
        request.identity(),
        request.source,
        request.topology.discretization_digest(),
        four.entity_counts(),
        remapped,
    )
    .unwrap();
    assert!(publish_spoof(&workload, &mesh, request, remapped, four).is_err());
}

// witness: FEA-EM-QUALIFICATION-001/accuracy @ magnetostatic/tet4/public_study; coverage: partial;
//   boundary: Sampled publication catalogs of 0.3 and 2.3 MB (2- and 4-cell slabs, beyond the
//   fixed per-contract allowance) are admitted by the accuracy reservation basis (four typed
//   decodes of their canonical bytes), and measured metadata validation stays within the
//   direct-encoding estimate; catalog sizing only, not field accuracy.
#[test]
fn large_catalog_metadata_validation_fits_the_typed_decode_reservation() {
    for cells in [2, 4] {
        let mesh = slab_mesh(cells);
        let (workload, recovery, fields, catalog, _) = fixture(&mesh);
        let ObservationFieldCatalogSource::SampledRecovery { sampling, .. } = catalog.source()
        else {
            panic!("sampled source")
        };
        let sampling = (**sampling).clone();
        // publish_spoof also measures the validation peak against the estimate.
        let publication = publish_spoof(&workload, &mesh, recovery, fields, sampling).unwrap();
        let bytes = publication.canonical_bytes().unwrap();
        let typed = runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&bytes)
            .unwrap();
        let required = publication.metadata_validation_resident_bytes().unwrap();
        eprintln!(
            "catalog cells={cells} encoded={} typed={typed} required={required}",
            bytes.len()
        );
        assert!(required <= 4 * typed, "{required} > 4 x {typed}");
    }
}
