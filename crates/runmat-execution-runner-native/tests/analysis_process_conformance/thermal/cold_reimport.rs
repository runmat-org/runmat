//! Cold import receives only the trusted root, authority and its declared physical closure.
//! This module never invokes a kernel or a driver execution entry point.
use super::super::publication_chain::MemoryStore;
use runmat_analysis_execution::{
    import_thermal_study_completion, import_thermal_study_root, prepare_thermal_study_root,
    CompletedThermalStudy, ThermalStudyResultClosure, ThermalStudyResultPhase,
};
use runmat_execution::Digest;
use runmat_execution_artifact::{cache::CacheImport, object::ObjectInventoryLimits};
use std::collections::BTreeMap;

pub(crate) fn assert_closed_completion(
    original: &impl CacheImport,
    completed: CompletedThermalStudy,
    limits: ObjectInventoryLimits,
) -> CompletedThermalStudy {
    assert_closed_completion_impl(original, completed, limits, true)
}

/// Full original closure/value replay with representative transport corruption tests.
/// The original generic lifecycle retains exhaustive per-object rejection coverage.
pub(crate) fn assert_closed_physical_completion(
    original: &impl CacheImport,
    completed: CompletedThermalStudy,
    limits: ObjectInventoryLimits,
) -> CompletedThermalStudy {
    assert_closed_completion_impl(original, completed, limits, false)
}

fn assert_closed_completion_impl(
    original: &impl CacheImport,
    completed: CompletedThermalStudy,
    limits: ObjectInventoryLimits,
    exhaustive: bool,
) -> CompletedThermalStudy {
    let prepared = prepare_thermal_study_root(&completed).unwrap();
    prepared.catalog.validate().unwrap();
    assert_eq!(prepared.catalog.roots, vec![prepared.root.clone()]);
    let access = completed.preparation().artifact_access;
    let trusted_root = prepared.root.logical_digest;
    let supplied: BTreeMap<_, _> = prepared
        .objects
        .into_iter()
        .map(|object| {
            object.validate().unwrap();
            (object.descriptor.digest, object.bytes)
        })
        .collect();
    let mut cold = MemoryStore::default();
    for reference in &prepared.catalog.objects {
        let bytes = supplied
            .get(&reference.logical_digest)
            .cloned()
            .unwrap_or_else(|| {
                original
                    .read_verified(reference.logical_digest)
                    .unwrap()
                    .expect("declared thermal child missing from producing store")
            });
        assert_eq!(Digest::sha256(&bytes), reference.logical_digest);
        assert_eq!(bytes.len() as u64, reference.encoded_length);
        assert!(cold.0.insert(reference.logical_digest, bytes).is_none());
    }
    assert_eq!(cold.0.len(), prepared.catalog.objects.len());
    drop(supplied);

    // Keep only expected semantic values for assertions, not the old coordinator or a store
    // fallback. The import below cannot consult the producing session/cache.
    let expected_result = completed.result.manifest.clone();
    let expected_model = completed.compiled.model.clone();
    let expected_workload = completed.workload.clone();
    let expected_scope = completed.preparation().scope;
    let expected_receipt = completed.result.numerical_receipt.clone();
    let expected_profiles = completed.result.state_publication_profiles.clone();
    let expected_accuracy = completed.accuracy.artifact.clone();
    let expected_states: Vec<_> = completed
        .states
        .iter()
        .map(|state| state.state.clone())
        .collect();
    let expected_baseline = completed.baseline_usage.evidence.clone();
    let expected_terminal = completed.terminal_usage.evidence.clone();
    let expected_publication = completed.publication.result.clone();
    let expected_catalog = completed.observations.catalog.clone();
    let expected_samples: Vec<_> = completed
        .observations
        .samples
        .iter()
        .map(|sample| sample.manifest.clone())
        .collect();
    let expected_histories: Vec<_> = completed
        .observations
        .histories
        .iter()
        .map(|history| history.publication.clone())
        .collect();
    let expected_closures = completed.result_closures().to_vec();
    drop(completed);

    let imported =
        import_thermal_study_root(&cold, &prepared.root, trusted_root, &access, limits).unwrap();
    assert_eq!(imported.result.manifest, expected_result);
    assert_eq!(imported.compiled.model, expected_model);
    assert_eq!(imported.workload, expected_workload);
    assert_eq!(imported.preparation().scope, expected_scope);
    assert_eq!(imported.result.numerical_receipt, expected_receipt);
    assert_eq!(
        imported.result.state_publication_profiles,
        expected_profiles
    );
    assert_eq!(imported.accuracy.artifact, expected_accuracy);
    assert_eq!(
        imported
            .states
            .iter()
            .map(|state| &state.state)
            .collect::<Vec<_>>(),
        expected_states.iter().collect::<Vec<_>>()
    );
    assert_eq!(imported.baseline_usage.evidence, expected_baseline);
    assert_eq!(imported.terminal_usage.evidence, expected_terminal);
    assert_eq!(imported.publication.result, expected_publication);
    assert_eq!(imported.observations.catalog, expected_catalog);
    assert_eq!(
        imported
            .observations
            .samples
            .iter()
            .map(|sample| &sample.manifest)
            .collect::<Vec<_>>(),
        expected_samples.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        imported
            .observations
            .histories
            .iter()
            .map(|history| &history.publication)
            .collect::<Vec<_>>(),
        expected_histories.iter().collect::<Vec<_>>()
    );
    assert_eq!(imported.result_closures(), expected_closures);
    let republished = prepare_thermal_study_root(&imported).unwrap();
    assert_eq!(republished.root, prepared.root);
    assert_eq!(republished.catalog, prepared.catalog);
    drop(republished);

    assert!(import_thermal_study_root(
        &cold,
        &prepared.root,
        Digest::sha256("foreign-thermal-root"),
        &access,
        limits
    )
    .is_err());
    let mut foreign_scope = access.clone();
    foreign_scope.authorization_scope.push_str("/foreign");
    assert!(
        import_thermal_study_root(&cold, &prepared.root, trusted_root, &foreign_scope, limits)
            .is_err()
    );
    let mut foreign_encryption = access.clone();
    foreign_encryption.encryption_context = Digest::sha256("foreign-thermal-encryption");
    assert!(import_thermal_study_root(
        &cold,
        &prepared.root,
        trusted_root,
        &foreign_encryption,
        limits
    )
    .is_err());
    let mut formats = std::collections::BTreeSet::new();
    let final_digest = prepared.catalog.objects.last().unwrap().logical_digest;
    for reference in &prepared.catalog.objects {
        // Each retained transport format, the trusted root and the final inventory entry.
        // This selects rejection witnesses only; every original object was imported and compared.
        let new_format = formats.insert((&reference.media_type, &reference.value_schema));
        if !exhaustive
            && !new_format
            && reference.logical_digest != trusted_root
            && reference.logical_digest != final_digest
        {
            continue;
        }
        let bytes = cold.0.remove(&reference.logical_digest).unwrap();
        assert!(
            import_thermal_study_root(&cold, &prepared.root, trusted_root, &access, limits)
                .is_err(),
            "missing thermal closure object {:?}",
            reference.logical_digest
        );
        let mut substituted = bytes.clone();
        substituted[0] ^= 1;
        cold.0.insert(reference.logical_digest, substituted);
        assert!(
            import_thermal_study_root(&cold, &prepared.root, trusted_root, &access, limits)
                .is_err(),
            "substituted thermal closure object {:?}",
            reference.logical_digest
        );
        cold.0.insert(reference.logical_digest, bytes);
    }
    assert_bad_transcripts(&cold, &imported);
    assert_foreign_contexts(&cold, &imported);
    imported
}

fn assert_bad_transcripts(cold: &MemoryStore, completed: &CompletedThermalStudy) {
    let valid = completed.result_closures();
    assert_observation_inventories(cold, completed);
    let reject = |case: &str, closures: Vec<ThermalStudyResultClosure>| {
        assert!(
            import_thermal_study_completion(cold, completed.preparation(), closures).is_err(),
            "accepted invalid thermal completion transcript: {case}"
        );
    };
    assert!(valid.len() > 4);
    let mut extra = valid.to_vec();
    extra.push(valid.last().unwrap().clone());
    reject("extra phase", extra);
    let mut omitted = valid.to_vec();
    omitted.remove(1);
    reject("omitted observation", omitted);
    let mut reordered = valid.to_vec();
    reordered.swap(1, 2); // Same Observation phase; exact task association must reject this.
    assert_eq!(valid[1].phase, ThermalStudyResultPhase::Observation);
    assert_eq!(valid[2].phase, ThermalStudyResultPhase::Observation);
    reject("reordered observations", reordered);
    let mut wrong_workload = valid.to_vec();
    wrong_workload[1].workload = valid[2].workload.clone();
    reject("foreign observation workload", wrong_workload);
    let mut wrong_response = valid.to_vec();
    wrong_response[1].response = valid[2].response.clone();
    reject("foreign observation response", wrong_response);
    let mut wrong_accounting = valid.to_vec();
    wrong_accounting[1].execution = valid[2].execution.clone();
    reject("foreign observation accounting", wrong_accounting);
    let mut wrong_phase = valid.to_vec();
    wrong_phase[0].phase = ThermalStudyResultPhase::Accuracy;
    reject("foreign phase label", wrong_phase);
    let mut wrong_state = valid.to_vec();
    let state = wrong_state
        .iter_mut()
        .find(|item| matches!(item.phase, ThermalStudyResultPhase::State { .. }))
        .unwrap();
    state.phase = ThermalStudyResultPhase::State {
        ordinal: usize::MAX,
    };
    reject("foreign state ordinal", wrong_state);
}

fn assert_foreign_contexts(cold: &MemoryStore, completed: &CompletedThermalStudy) {
    use runmat_analysis_execution::ThermalStudyScope;
    use runmat_analysis_fea::study::{
        plan_thermal_step, plan_thermal_step_with_fields, ThermalStudyPlanningPolicy,
    };
    let policy = ThermalStudyPlanningPolicy {
        backend: completed.workload.backend,
        element_heat_source: None,
    };
    let mut foreign_model = completed.preparation();
    foreign_model.product.compiled.model.materials[0]
        .thermal
        .as_mut()
        .unwrap()
        .expansion_coefficient_per_k = runmat_types::physical::Quantity::new(
        foreign_model.product.compiled.model.materials[0]
            .thermal
            .as_mut()
            .unwrap()
            .expansion_coefficient_per_k
            .canonical_value()
            + (1.0e-5),
    );
    foreign_model.product.compiled.properties =
        runmat_analysis_fea::properties::compile_property_map(
            &foreign_model.product.compiled.model,
            Some(&foreign_model.mesh),
        )
        .unwrap();
    foreign_model.product.workload = plan_thermal_step_with_fields(
        &foreign_model.product.compiled,
        completed.workload.step_id.as_str(),
        policy.clone(),
        completed.workload.initial_fields.clone(),
    )
    .unwrap();
    assert!(
        import_thermal_study_completion(cold, foreign_model, completed.result_closures().to_vec())
            .is_err(),
        "saved thermal transcript cannot be reused under a foreign full material model"
    );
    if completed.preparation().scope == ThermalStudyScope::SelectedStep {
        let mut foreign_step = completed.preparation();
        foreign_step.product.workload = plan_thermal_step(
            &foreign_step.product.compiled,
            "unselected_heat",
            policy.clone(),
        )
        .unwrap();
        assert!(
            import_thermal_study_completion(
                cold,
                foreign_step,
                completed.result_closures().to_vec()
            )
            .is_err(),
            "valid alternate thermal step cannot consume the selected heat transcript"
        );
        let mut standalone = completed.preparation();
        standalone.scope = ThermalStudyScope::Standalone;
        assert!(
            import_thermal_study_completion(cold, standalone, completed.result_closures().to_vec())
                .is_err(),
            "mixed-model selected scope cannot be downgraded to standalone"
        );
    } else {
        // Explicit selection of the sole step is physically identical. Scope is a root
        // contract distinction, not an artificial change to the numerical task identity.
        let original_root = prepare_thermal_study_root(completed).unwrap();
        let mut selected = completed.preparation();
        selected.scope = ThermalStudyScope::SelectedStep;
        let selected =
            import_thermal_study_completion(cold, selected, completed.result_closures().to_vec())
                .unwrap();
        assert_eq!(selected.result.manifest, completed.result.manifest);
        assert_ne!(
            prepare_thermal_study_root(&selected).unwrap().root,
            original_root.root
        );
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InventoryMutation {
    Reordered,
    OmittedChild,
    ExtraChild,
}

fn assert_observation_inventories(cold: &MemoryStore, completed: &CompletedThermalStudy) {
    use runmat_execution::value::ValuePayload;
    use runmat_execution_artifact::ProgramExecutionResponse;

    let valid = completed.result_closures();
    for ordinal in [1, 2] {
        assert_eq!(valid[ordinal].phase, ThermalStudyResultPhase::Observation);
        for mutation in [
            InventoryMutation::Reordered,
            InventoryMutation::OmittedChild,
            InventoryMutation::ExtraChild,
        ] {
            let mut transcript = valid.to_vec();
            let closure = &mut transcript[ordinal];
            let ProgramExecutionResponse::ExternalizedSuccess {
                outputs,
                result_objects,
            } = &mut closure.response
            else {
                panic!("thermal observation requires externalized objects");
            };
            assert!(result_objects.len() > 1);
            match mutation {
                InventoryMutation::Reordered => result_objects.swap(0, 1),
                InventoryMutation::OmittedChild => {
                    let child = result_objects
                        .iter()
                        .position(|reference| {
                            !outputs.iter().any(|output| {
                                matches!(output,
                            ValuePayload::Object(root) if root.as_ref() == reference)
                            })
                        })
                        .expect("observation has a non-root physical child");
                    result_objects.remove(child);
                }
                InventoryMutation::ExtraChild => {
                    let foreign = valid
                        .iter()
                        .filter_map(|phase| match &phase.response {
                            ProgramExecutionResponse::ExternalizedSuccess {
                                result_objects,
                                ..
                            } => Some(result_objects),
                            _ => None,
                        })
                        .flatten()
                        .filter(|reference| !result_objects.contains(reference))
                        .min_by_key(|reference| reference.encoded_length)
                        .expect("another phase supplies an unrelated object")
                        .clone();
                    result_objects.push(foreign);
                }
            }
            let artifact_bytes = result_objects
                .iter()
                .try_fold(0_u64, |bytes, reference| {
                    bytes.checked_add(reference.encoded_length)
                })
                .unwrap();
            let original = &closure.execution.winning_accounting().resources;
            if mutation == InventoryMutation::Reordered {
                assert_eq!(artifact_bytes, original.artifact_bytes);
            }
            // The original immutable commit cannot be rewritten to match a forged
            // inventory. Association rejects the altered response before physical import.
            let error =
                match import_thermal_study_completion(cold, completed.preparation(), transcript) {
                    Ok(_) => panic!("accepted {mutation:?} observation {ordinal} inventory"),
                    Err(error) => error,
                };
            assert!(
                error
                    .to_string()
                    .contains("phase response or callable differs from its original task"),
                "wrong rejection for {mutation:?} observation {ordinal}: {error}"
            );
        }
    }
}
