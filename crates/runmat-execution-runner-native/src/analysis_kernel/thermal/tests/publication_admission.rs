use super::*;
use runmat_analysis_execution::{
    prepare_semantic_state_publication, PreparedSemanticStatePublication,
    SemanticStatePublicationPreparation,
};
use runmat_analysis_fea::execution::{SemanticStatePublicationSource, StatePublicationProfiles};

fn publication(order: ElementOrder, underdeclare: bool) -> PreparedSemanticStatePublication {
    let (workload, task) = task(order, 1024);
    let outputs = NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::SolveThermal,
            host: &task.host,
            inputs: &task.inputs,
            output_policy: policy(1024),
        })
        .unwrap();
    let mut outputs = outputs.into_iter().map(|output| {
        output
            .into_prepared(1024, ObjectInventoryLimits::default())
            .unwrap()
    });
    let result = outputs.next().unwrap();
    let _receipt = outputs.next().unwrap();
    let mut profiles = outputs.next().unwrap();
    let manifest = decode_prepared_thermal_manifest(&result).unwrap();
    if underdeclare {
        let catalog = StatePublicationProfiles::from_canonical_bytes(&profiles.payload()).unwrap();
        let mut entries = catalog.entries().to_vec();
        entries[0].encoded_bytes_upper_bound -= 1;
        let forged = StatePublicationProfiles::new(
            manifest.producer,
            manifest.identity(),
            manifest.topology_digest,
            result.manifest.payload_digest,
            entries,
        )
        .unwrap();
        profiles = prepare_analysis_objects(
            AnalysisNumericalObjectKind::StatePublicationProfiles,
            forged.identity(),
            &forged.canonical_bytes().unwrap(),
            1024,
            ObjectNamespace::ResultValue,
            ObjectInventoryLimits::default(),
        )
        .unwrap();
    }
    prepare_semantic_state_publication(SemanticStatePublicationPreparation {
        source_host: task.host,
        source_result: result,
        source_profiles: profiles,
        source: SemanticStatePublicationSource::ThermalPrimaryField,
        declaration: workload.model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: manifest.topology_digest,
        program_revision: revision(),
        chunk_bytes: 1024,
        inventory_limits: ObjectInventoryLimits::default(),
    })
    .unwrap()
}

#[test]
fn publication_worker_rederives_source_profile_despite_self_consistent_forged_plan() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let forged = publication(order, true);
        forged.resource_plan.validate_identity().unwrap();
        let error = NativeAnalysisKernel
            .execute(AnalysisKernelInvocation {
                kind: AnalysisWorkKind::PublishState,
                host: &forged.host,
                inputs: &forged.inputs,
                output_policy: policy(1024),
            })
            .unwrap_err();
        assert!(
            matches!(
                error,
                AnalysisBridgeError::Identity(
                    "state publication profile differs from its numerical source"
                )
            ),
            "{error:?}"
        );
    }
}

#[test]
fn publication_rejects_missing_substituted_and_duplicate_admission_inputs() {
    let task = publication(ElementOrder::Tet4, false);
    let other = publication(ElementOrder::Tet10, false);
    let invoke = |inputs: &[runmat_analysis_execution::PreparedAnalysisObjects]| {
        NativeAnalysisKernel.execute(AnalysisKernelInvocation {
            kind: AnalysisWorkKind::PublishState,
            host: &task.host,
            inputs,
            output_policy: policy(1024),
        })
    };
    assert!(invoke(&task.inputs).is_ok());
    for index in 0..task.inputs.len() {
        let mut inputs = task.inputs.clone();
        inputs.remove(index);
        assert!(invoke(&inputs).is_err());
        let mut inputs = task.inputs.clone();
        inputs[index] = other.inputs[index].clone();
        assert!(invoke(&inputs).is_err(), "substituted input {index}");
    }
    let mut inputs = task.inputs.clone();
    inputs.push(inputs[0].clone());
    assert!(invoke(&inputs).is_err());
}
