//! A composed prefix is replayed and bound to its own preload, never just colocated.
use super::*;
use runmat_analysis_execution::{
    prepare_structural_sequence_step, CompletedThermalStudy, StructuralLinearProgramPreparation,
    ThermalPrestressSource,
};
use runmat_analysis_fea::execution::{
    CanonicalSemanticState, SemanticStatePublicationRequest, SemanticStatePublicationSource,
    StructuralLinearStateInput, StructuralLinearWorkloadResult,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ThermalPrefixClosure {
    pub root: ValueRef,
    pub catalog: PublishedObjectCatalog,
}

impl ThermalPrefixClosure {
    pub(super) fn validate(&self, access: &ArtifactAccess) -> NativeExecutionResult<()> {
        self.catalog.validate().map_err(protocol)?;
        if self.catalog.roots != vec![self.root.clone()] {
            return Err(protocol(
                "thermal prefix must declare exactly its completion root",
            ));
        }
        for reference in &self.catalog.objects {
            if reference.kind != ValueRefKind::ResultObject
                || reference.authorization_scope != access.authorization_scope
                || reference.encryption_context != access.encryption_context
            {
                return Err(protocol(
                    "thermal prefix object is outside result authority",
                ));
            }
        }
        Ok(())
    }
}

pub(super) struct ImportContext<'a, S> {
    pub source: &'a S,
    pub compiled: Option<&'a runmat_analysis_fea::study::CompiledAnalysisStudy>,
    pub analysis: &'a ExactStructuralStudyAnalysisClosure,
    pub thermal: Option<&'a ThermalPrefixClosure>,
    pub expected: &'a ExactStructuralStudyResultExpectation,
    pub workloads: &'a StructuralStudyWorkloads,
    pub mesh: &'a runmat_meshing_core::SolverMeshArtifact,
    pub limits: ObjectInventoryLimits,
}

pub(super) fn import<S: CacheImport>(
    context: ImportContext<'_, S>,
) -> NativeExecutionResult<Option<Box<CompletedThermalStudy>>> {
    let ImportContext {
        source,
        compiled,
        analysis,
        thermal,
        expected,
        workloads,
        mesh,
        limits,
    } = context;
    let prefix = if matches!(
        workloads,
        StructuralStudyWorkloads::HeterogeneousSequence { .. }
    ) {
        workloads.prestress_prefix().ok()
    } else {
        None
    };
    let thermal_workload = prefix.as_ref().and_then(|prefix| prefix.thermal);
    let (Some(thermal_workload), Some(closure)) = (thermal_workload, thermal) else {
        if thermal_workload.is_some() || thermal.is_some() {
            return Err(protocol(
                "thermal root and authored prestress prefix must occur together",
            ));
        }
        return Ok(None);
    };
    closure.validate(&expected.artifact_access)?;
    let recovered = runmat_analysis_execution::import_thermal_study_root(
        source,
        &closure.root,
        closure.root.logical_digest,
        &expected.artifact_access,
        limits,
    )
    .map_err(protocol)?;
    let replay = ThermalPrestressSource::import(
        source,
        recovered.preparation(),
        recovered.result_closures().to_vec(),
    )
    .map_err(protocol)?;
    let completed = replay.completed();
    let recovered_catalog =
        runmat_analysis_execution::prepare_thermal_study_root(completed).map_err(protocol)?;
    if recovered_catalog.root != closure.root || recovered_catalog.catalog != closure.catalog {
        return Err(protocol(
            "composed thermal inventory differs from its replayed closure",
        ));
    }
    let mut compiled = compiled
        .cloned()
        .ok_or_else(|| protocol("thermal composition omits original compiled definition"))?;
    // Input materializations and region aliases are replayed by the containing importer.
    compiled.model = workloads.authored_model().map_err(protocol)?.clone();
    compiled.validate_against(mesh).map_err(protocol)?;
    let preparation = completed.preparation();
    if completed.compiled != compiled
        || &completed.workload != thermal_workload
        || preparation.mesh != *mesh
        || preparation.artifact_access != expected.artifact_access
        || preparation.program_revision != expected.program_revision
        || preparation.frozen_project_digest != expected.frozen_project_digest
    {
        return Err(protocol(
            "thermal prefix differs from containing compiled execution context",
        ));
    }
    let preload = prefix.as_ref().expect("thermal prefix").preload;
    let [input] = preload
        .procedure_step()
        .map_err(protocol)?
        .state_inputs
        .as_slice()
    else {
        return Err(protocol("thermal preload requires one named input"));
    };
    let runmat_analysis_core::StateSource::Upstream { state_id, .. } = &input.source else {
        return Err(protocol("thermal preload requires an upstream state"));
    };
    let state = completed
        .states
        .iter()
        .find(|state| &state.state.state_id == state_id)
        .ok_or_else(|| protocol("composed root did not publish the preload's named state"))?;
    let prepared = prepare_structural_sequence_step(StructuralLinearProgramPreparation {
        authored_study_digest: compiled.definition_digest().map_err(protocol)?,
        frozen_project_digest: expected.frozen_project_digest,
        program_revision: expected.program_revision.clone(),
        artifact_access: expected.artifact_access.clone(),
        workload: preload.clone(),
        mesh: Some(mesh.clone()),
        state_input: Some(state.objects.clone()),
        chunk_bytes: preparation.output_policy.chunk_bytes(),
        inventory_limits: preparation.output_policy.inventory_limits(),
        required_capabilities: preparation.required_capabilities.clone(),
        accelerator_requests: expected.analysis_accelerator_requests.clone(),
        local_execution: expected.local_execution,
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .map_err(protocol)?;
    let (result_root, receipt_root, state_root, objects) = match analysis {
        ExactStructuralStudyAnalysisClosure::StructuralBuckling {
            preload_result_root,
            preload_receipt_root,
            prestress_state_root,
            objects,
            ..
        }
        | ExactStructuralStudyAnalysisClosure::StructuralPrestressedModal {
            preload_result_root,
            preload_receipt_root,
            prestress_state_root,
            objects,
            ..
        }
        | ExactStructuralStudyAnalysisClosure::StructuralPrestressedFrequency {
            preload_result_root,
            preload_receipt_root,
            prestress_state_root,
            objects,
            ..
        }
        | ExactStructuralStudyAnalysisClosure::StructuralPrestressedModalFrequency {
            preload_result_root,
            preload_receipt_root,
            prestress_state_root,
            objects,
            ..
        } => (
            preload_result_root,
            preload_receipt_root,
            prestress_state_root,
            objects,
        ),
        _ => {
            return Err(protocol(
                "thermal prefix has a non-prestress structural result",
            ))
        }
    };
    let load = |root: &ValueRef, kind| {
        result_import::import_numerical_analysis_objects(
            source,
            root,
            objects,
            kind,
            &expected.artifact_access,
            limits,
        )
    };
    let result_objects = load(result_root, AnalysisNumericalObjectKind::SolverResult)?;
    let receipt_objects = load(receipt_root, AnalysisNumericalObjectKind::NumericalReceipt)?;
    let result = StructuralLinearWorkloadResult::from_canonical_bytes(&result_objects.payload())
        .map_err(protocol)?;
    let receipt = runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::from_canonical_bytes(&receipt_objects.payload()).map_err(protocol)?;
    let result_identity = result
        .result_identity(prepared.host.workload.identity)
        .map_err(protocol)?;
    receipt
        .validate_against(result_identity, &result)
        .map_err(protocol)?;
    if receipt.source_workload != prepared.host.workload.identity
        || result_objects.manifest.logical_identity != result_identity.digest()
        || receipt_objects.manifest.logical_identity != receipt.identity().digest()
    {
        return Err(protocol(
            "thermal preload receipt belongs to another prepared source workload",
        ));
    }
    let linearization = result
        .linearization
        .as_ref()
        .ok_or_else(|| protocol("thermal preload omitted linearization"))?;
    let retained = linearization
        .thermal_source()
        .ok_or_else(|| protocol("thermal preload omitted temperature lineage"))?;
    retained.validate_model(&preload.model).map_err(protocol)?;
    let invocation = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverInvocation)
        .expect("prepared invocation");
    let bound =
        StructuralLinearWorkload::from_canonical_bytes(&invocation.payload()).map_err(protocol)?;
    let Some(StructuralLinearStateInput::ThermalPrestress(transition)) = bound.state_input else {
        return Err(protocol(
            "composed preload is not bound to thermal prestress",
        ));
    };
    if retained.transition() != &transition || retained.structural_result() != result_identity {
        return Err(protocol(
            "structural result did not consume this root's thermal state",
        ));
    }
    let published_state = runmat_analysis_execution::decode_prepared_semantic_state(&load(
        state_root,
        AnalysisNumericalObjectKind::SemanticState,
    )?)
    .map_err(protocol)?;
    let declaration = preload
        .procedure_step()
        .map_err(protocol)?
        .state_outputs
        .iter()
        .find(|output| output.kind == runmat_analysis_core::StateKind::GeometricStiffness)
        .ok_or_else(|| protocol("thermal preload lacks geometric state declaration"))?;
    let request = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralLinearization,
        prepared.host.workload.step,
        declaration.clone(),
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        result_identity,
    )
    .map_err(protocol)?;
    let expected_state =
        CanonicalSemanticState::from_linearization_publication(&request, linearization.clone())
            .map_err(protocol)?;
    if published_state != expected_state {
        return Err(protocol(
            "geometric state differs from this root's thermal preload",
        ));
    }
    Ok(Some(Box::new(completed.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (ArtifactAccess, ThermalPrefixClosure) {
        let access = ArtifactAccess {
            authorization_scope: "composed-thermal-root-test".into(),
            encryption_context: Digest::sha256(b"composed-thermal-root-encryption"),
        };
        let reference = |bytes: &[u8]| {
            let object = LogicalObject::new(
                ObjectNamespace::ResultValue,
                "analysis/test-composed-thermal-root",
                "application/vnd.runmat.test",
                bytes.to_vec(),
            )
            .unwrap();
            access
                .reference(&object, ValueRefKind::ResultObject, "runmat.test")
                .unwrap()
        };
        let root = reference(b"thermal-root");
        let descendant = reference(b"thermal-descendant");
        let catalog =
            PublishedObjectCatalog::new(vec![root.clone()], vec![root.clone(), descendant])
                .unwrap();
        (access, ThermalPrefixClosure { root, catalog })
    }

    #[test]
    fn thermal_prefix_requires_its_exact_root_and_containing_authority() {
        let (access, closure) = fixture();
        closure.validate(&access).unwrap();
        let mut changed = closure.clone();
        changed.root = changed
            .catalog
            .objects
            .iter()
            .find(|object| object != &&closure.root)
            .unwrap()
            .clone();
        assert!(changed.validate(&access).is_err());
        let mut other_access = access;
        other_access.authorization_scope.push_str("-other");
        assert!(closure.validate(&other_access).is_err());
    }

    #[test]
    fn composed_inventory_deduplicates_shared_descendants_without_weakening_references() {
        let (_, closure) = fixture();
        let shared = closure.catalog.objects[0].clone();
        let merged = unique_references(
            closure
                .catalog
                .objects
                .iter()
                .cloned()
                .chain(std::iter::once(shared.clone())),
        )
        .unwrap();
        assert_eq!(merged, closure.catalog.objects);
        reference_inventory(&merged).unwrap();
        let mut substituted = shared;
        substituted.encoded_length += 1;
        assert!(unique_references(
            closure
                .catalog
                .objects
                .into_iter()
                .chain(std::iter::once(substituted))
        )
        .is_err());
    }
}
