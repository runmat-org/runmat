use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
    StructuralFrequencyEntryRequest, StructuralFrequencyEntryResult,
    StructuralFrequencyResultAssemblyRequest, StructuralFrequencySolverPlan,
    StructuralFrequencyStateInputs, StructuralFrequencyWorkload,
};
use runmat_analysis_fea::planning::resource::{
    build_structural_frequency_result_assembly_resource_plan,
    structural_frequency_entry_payload_bytes_upper_bound,
    structural_frequency_receipt_payload_bytes_upper_bound,
    structural_frequency_result_payload_bytes_upper_bound, FeaResourcePhase, FeaResourcePlan,
    PreparedPhaseResourceFacts, StructuralFrequencyEntryResourceProjection,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ObjectNamespace;
use runmat_execution_artifact::{ArtifactAccess, ProgramExecutionResponse};
use runmat_meshing_core::CanonicalMeshingContract;

use crate::{
    decode_prepared_semantic_state, decode_prepared_structural_frequency_entry_request,
    decode_prepared_structural_frequency_entry_result,
    decode_prepared_structural_frequency_result_assembly_request,
    decode_prepared_structural_frequency_workload, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_program, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralFrequencyEntryPreparation {
    pub output_policy: crate::AnalysisOutputPolicy,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub request: PreparedAnalysisObjects,
    pub prestress_state: Option<PreparedAnalysisObjects>,
    pub modal_state: Option<PreparedAnalysisObjects>,
    pub resource_plan: PreparedAnalysisObjects,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
}

pub struct StructuralFrequencyFanoutPreparation {
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub prestress_state: Option<PreparedAnalysisObjects>,
    pub modal_state: Option<PreparedAnalysisObjects>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralFrequencyEntryTask {
    pub workload: StructuralFrequencyWorkload,
    pub request: StructuralFrequencyEntryRequest,
    pub prestress_state: Option<CanonicalSemanticState>,
    pub modal_state: Option<CanonicalSemanticState>,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

pub fn prepare_structural_frequency_fanout(
    preparation: StructuralFrequencyFanoutPreparation,
) -> AnalysisBridgeResult<Vec<PreparedStructuralFrequencyEntryTask>> {
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency fan-out requires bounded nonzero chunks".into(),
        ));
    }
    let workload = decode_prepared_structural_frequency_workload(&preparation.workload)?;
    if preparation.mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency fan-out requires one canonical mesh object".into(),
        ));
    }
    let mesh =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&preparation.mesh.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let prestress_state = preparation
        .prestress_state
        .as_ref()
        .map(decode_prepared_semantic_state)
        .transpose()?;
    let prepared = match prestress_state.as_ref() {
        Some(state) => runmat_analysis_fea::pipeline::structural_frequency::prepare_prestressed_structural_frequency(
            &workload, &mesh, state,
        )?,
        None => runmat_analysis_fea::pipeline::structural_frequency::prepare_structural_frequency(
            &workload, &mesh,
        )?,
    };
    let modal_state = preparation
        .modal_state
        .as_ref()
        .map(decode_prepared_semantic_state)
        .transpose()?;
    let compiled =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            workload.model(),
            Some(&mesh),
            workload.procedure_step()?.accuracy.resources.resident_byte_budget(),
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_projection = StructuralFrequencyEntryResourceProjection::new_from_compilation(
        &mesh,
        &compiled,
        &workload,
        StructuralFrequencyStateInputs {
            prestress: prestress_state.as_ref(),
            modal_basis: modal_state.as_ref(),
        },
        &[],
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    workload
        .plan()
        .entries()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
        .into_iter()
        .map(|entry| {
            let request_value = StructuralFrequencyEntryRequest::new(
                &workload,
                prepared.study,
                prepared.step,
                prepared.execution_plan,
                prepared.scope.clone(),
                entry,
            )?;
            let output = crate::projected_output_resource_facts(
                structural_frequency_entry_payload_bytes_upper_bound(&request_value)
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
            )?;
            let request = crate::prepare_structural_frequency_entry_request_objects(
                &request_value,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )?;
            let mut phase_inputs = vec![
                preparation.workload.clone(),
                preparation.mesh.clone(),
                request.clone(),
            ];
            phase_inputs.extend(preparation.prestress_state.iter().cloned());
            phase_inputs.extend(preparation.modal_state.iter().cloned());
            let resource_plan = resource_projection
                .plan_for(
                    &request_value,
                    crate::prepared_input_resource_facts(&phase_inputs)?,
                    output,
                    crate::projected_output_set_resource_facts(),
                    preparation.local_execution,
                )
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            let resource_plan_objects = crate::prepare_analysis_objects(
                AnalysisNumericalObjectKind::ResourcePlan,
                resource_plan.identity.digest(),
                &resource_plan
                    .canonical_bytes()
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                preparation.chunk_bytes,
                ObjectNamespace::InputValue,
                preparation.inventory_limits,
            )?;
            prepare_structural_frequency_entry_task_with_admitted_resource_plan(
                StructuralFrequencyEntryPreparation {
                    output_policy: crate::AnalysisOutputPolicy::new(
                        preparation.chunk_bytes,
                        preparation.inventory_limits,
                    )?,
                    workload: preparation.workload.clone(),
                    mesh: preparation.mesh.clone(),
                    request,
                    prestress_state: preparation.prestress_state.clone(),
                    modal_state: preparation.modal_state.clone(),
                    resource_plan: resource_plan_objects,
                    artifact_access: preparation.artifact_access.clone(),
                    program_revision: preparation.program_revision.clone(),
                },
                Some(resource_plan),
            )
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralFrequencyEntry {
    pub result: StructuralFrequencyEntryResult,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_frequency_entry_task(
    preparation: StructuralFrequencyEntryPreparation,
) -> AnalysisBridgeResult<PreparedStructuralFrequencyEntryTask> {
    prepare_structural_frequency_entry_task_with_admitted_resource_plan(preparation, None)
}

fn prepare_structural_frequency_entry_task_with_admitted_resource_plan(
    preparation: StructuralFrequencyEntryPreparation,
    admitted_resource_plan: Option<FeaResourcePlan>,
) -> AnalysisBridgeResult<PreparedStructuralFrequencyEntryTask> {
    let workload = decode_prepared_structural_frequency_workload(&preparation.workload)?;
    let request = decode_prepared_structural_frequency_entry_request(&preparation.request)?;
    if preparation.mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency entry requires one canonical mesh object".into(),
        ));
    }
    let mesh =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&preparation.mesh.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let modal_state = preparation
        .modal_state
        .as_ref()
        .map(decode_prepared_semantic_state)
        .transpose()?;
    let prestress_state = preparation
        .prestress_state
        .as_ref()
        .map(decode_prepared_semantic_state)
        .transpose()?;
    if preparation.resource_plan.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency entry requires one canonical resource plan".into(),
        ));
    }
    let resource_plan = FeaResourcePlan::from_canonical_bytes(&preparation.resource_plan.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let modal_state_matches = match (workload.solver(), modal_state.as_ref()) {
        (StructuralFrequencySolverPlan::Direct { .. }, None) => true,
        (StructuralFrequencySolverPlan::ModalReduced { basis_state }, Some(state)) => {
            state.identity == *basis_state
        }
        _ => false,
    };
    let state_topology_matches = prestress_state
        .iter()
        .chain(modal_state.iter())
        .all(|state| {
            state.topology_digest
                == runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        });
    let prestress_state_matches = match (workload.prestress_transition(), prestress_state.as_ref())
    {
        (None, None) => true,
        (Some(transition), Some(state)) => state.identity == transition.source_state,
        _ => false,
    };
    if preparation.workload.manifest.logical_identity != workload.identity().digest()
        || preparation.mesh.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
        || !modal_state_matches
        || !prestress_state_matches
        || !state_topology_matches
        || preparation.resource_plan.manifest.logical_identity != resource_plan.identity.digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency inputs differ from their canonical identities",
        ));
    }
    if resource_plan.input.phase != FeaResourcePhase::StructuralFrequencyEntry(request.identity())
        || !resource_plan.request.accelerators.is_empty()
    {
        return Err(AnalysisBridgeError::Identity(
            "frequency entry task received a resource plan for another task",
        ));
    }
    let expected_resource_plan = if let Some(admitted_resource_plan) = admitted_resource_plan {
        admitted_resource_plan
    } else {
        match prestress_state.as_ref() {
            Some(state) => runmat_analysis_fea::pipeline::structural_frequency::validate_prestressed_structural_frequency_entry(
                &workload, &mesh, &request, state,
            )?,
            None => runmat_analysis_fea::pipeline::structural_frequency::validate_structural_frequency_entry(
                &workload, &mesh, &request,
            )?,
        };
        let constraints =
            runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
                workload.model(),
                Some(&mesh),
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
        let mut phase_inputs = vec![
            preparation.workload.clone(),
            preparation.mesh.clone(),
            preparation.request.clone(),
        ];
        phase_inputs.extend(preparation.prestress_state.iter().cloned());
        phase_inputs.extend(preparation.modal_state.iter().cloned());
        runmat_analysis_fea::planning::resource::build_structural_frequency_entry_resource_plan(
            &mesh,
            &constraints,
            &workload,
            &request,
            StructuralFrequencyStateInputs {
                prestress: prestress_state.as_ref(),
                modal_basis: modal_state.as_ref(),
            },
            &resource_plan.request.accelerators,
            PreparedPhaseResourceFacts {
                inputs: crate::prepared_input_resource_facts(&phase_inputs)?,
                output: crate::projected_output_resource_facts(
                    structural_frequency_entry_payload_bytes_upper_bound(&request)
                        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
                )?,
                output_set: crate::projected_output_set_resource_facts(),
            },
            runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                resource_plan.request.cpu_millicores,
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        )
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?
    };
    if resource_plan != expected_resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "frequency entry resource plan does not converge from authoritative inputs",
        ));
    }
    let resource_plan_objects = preparation.resource_plan.clone();
    let mut typed_inputs = vec![
        (preparation.workload, ValueRefKind::DriverObject),
        (preparation.mesh, ValueRefKind::DriverObject),
        (preparation.request, ValueRefKind::DriverObject),
        (preparation.resource_plan, ValueRefKind::DriverObject),
    ];
    if let Some(state) = preparation.prestress_state {
        typed_inputs.push((state, ValueRefKind::ResultObject));
    }
    if let Some(state) = preparation.modal_state {
        typed_inputs.push((state, ValueRefKind::ResultObject));
    }
    let workload_inputs = typed_inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let inputs = typed_inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let analysis_workload = AnalysisWorkloadRequest::new(
        request.study(),
        request.step(),
        request.execution_plan(),
        AnalysisWorkKind::SolveStructuralFrequencyEntry,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::StructuralFrequencyEntryResult],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        analysis_workload,
        preparation.artifact_access,
        preparation.output_policy,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralFrequencyEntryTask {
        workload,
        request,
        prestress_state,
        modal_state,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_frequency_entry(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralFrequencyEntryTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralFrequencyEntry> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency entry must return exactly one result".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::StructuralFrequencyEntryResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = decode_prepared_structural_frequency_entry_result(&objects)?;
    if result.request() != &prepared.request {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency result differs from its prepared request",
        ));
    }
    Ok(ImportedStructuralFrequencyEntry { result, objects })
}

pub struct StructuralFrequencyResultPreparation {
    pub request: PreparedAnalysisObjects,
    pub entries: Vec<PreparedAnalysisObjects>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralFrequencyResultTask {
    pub request: StructuralFrequencyResultAssemblyRequest,
    pub entries: Vec<StructuralFrequencyEntryResult>,
    pub resource_plan: FeaResourcePlan,
    pub resource_plan_objects: PreparedAnalysisObjects,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralFrequencyResult {
    pub result: runmat_analysis_fea::execution::StructuralFrequencySolverResult,
    pub numerical_receipt: runmat_analysis_fea::execution::StructuralFrequencyNumericalReceipt,
    pub result_objects: PreparedAnalysisObjects,
    pub receipt_objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_structural_frequency_result_task(
    preparation: StructuralFrequencyResultPreparation,
) -> AnalysisBridgeResult<PreparedStructuralFrequencyResultTask> {
    if preparation.chunk_bytes == 0 {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency assembly requires bounded nonzero chunks".into(),
        ));
    }
    let request =
        decode_prepared_structural_frequency_result_assembly_request(&preparation.request)?;
    let entries = preparation
        .entries
        .iter()
        .map(decode_prepared_structural_frequency_entry_result)
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    request.validate_for(&entries)?;
    let first = entries.first().ok_or_else(|| {
        AnalysisBridgeError::Invalid("structural frequency assembly has no entries".into())
    })?;
    if entries.iter().any(|entry| {
        entry.request().study() != first.request().study()
            || entry.request().step() != first.request().step()
            || entry.request().execution_plan() != first.request().execution_plan()
    }) {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency entries have inconsistent task authority",
        ));
    }
    let mut numerical_inputs = vec![preparation.request.clone()];
    numerical_inputs.extend(preparation.entries.iter().cloned());
    let input_facts = crate::prepared_input_resource_facts(&numerical_inputs)?;
    let output_record_count = u64::try_from(entries.len())
        .map_err(|_| AnalysisBridgeError::Invalid("frequency entry count overflows".into()))?
        .checked_add(1)
        .ok_or_else(|| AnalysisBridgeError::Invalid("frequency output count overflows".into()))?;
    let result_payload_upper_bound =
        structural_frequency_result_payload_bytes_upper_bound(&input_facts, output_record_count)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let receipt_payload_upper_bound =
        structural_frequency_receipt_payload_bytes_upper_bound(&input_facts, output_record_count)
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan = build_structural_frequency_result_assembly_resource_plan(
        &request,
        &entries,
        input_facts,
        crate::projected_output_resource_facts(result_payload_upper_bound)?,
        crate::projected_output_resource_facts(receipt_payload_upper_bound)?,
        crate::projected_output_set_resource_facts(),
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let resource_plan_objects = crate::prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan
            .canonical_bytes()
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut typed_inputs = vec![
        (preparation.request, ValueRefKind::DriverObject),
        (resource_plan_objects.clone(), ValueRefKind::DriverObject),
    ];
    typed_inputs.extend(
        preparation
            .entries
            .into_iter()
            .map(|entry| (entry, ValueRefKind::ResultObject)),
    );
    let workload_inputs = typed_inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let inputs = typed_inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let analysis_workload = AnalysisWorkloadRequest::new(
        first.request().study(),
        first.request().step(),
        first.request().execution_plan(),
        AnalysisWorkKind::AssembleStructuralFrequencyResult,
        workload_inputs,
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(
        analysis_workload,
        preparation.artifact_access,
        crate::AnalysisOutputPolicy::new(preparation.chunk_bytes, preparation.inventory_limits)?,
    )?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralFrequencyResultTask {
        request,
        entries,
        resource_plan,
        resource_plan_objects,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_frequency_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralFrequencyResultTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralFrequencyResult> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response.clone(), &prepared.host)?;
    let [result_output, receipt_output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "structural frequency assembly must return result and receipt roots".into(),
        ));
    };
    let result_objects = import_analysis_objects(
        source,
        &result_output.root,
        AnalysisNumericalObjectKind::SolverResult,
        result_output.logical_identity,
        result_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let receipt_objects = import_analysis_objects(
        source,
        &receipt_output.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt_output.logical_identity,
        receipt_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = crate::decode_prepared_structural_frequency_solver_result(&result_objects)?;
    let numerical_receipt =
        crate::decode_prepared_structural_frequency_numerical_receipt(&receipt_objects)?;
    numerical_receipt.validate_for_result(&result)?;
    let expected = prepared.request.assemble(&prepared.entries)?;
    if result != expected.result || numerical_receipt != expected.receipt {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency outputs differ from the prepared fan-in",
        ));
    }
    let mut expected = result_objects
        .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "structural frequency result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralFrequencyResult {
        result,
        numerical_receipt,
        result_objects,
        receipt_objects,
        response,
    })
}
