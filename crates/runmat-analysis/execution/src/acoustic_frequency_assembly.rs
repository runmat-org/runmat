//! Admitted immutable fan-in of original acoustic frequency results.
#[cfg(test)]
mod tests;
use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisOutputPolicy,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};
use runmat_analysis_fea::execution::{
    AcousticFrequencyResult, AcousticFrequencyWorkload, AcousticHarmonicSolverResult,
    AnalysisNumericalObjectKind, AnalysisWorkKind, ExecutionPlanIdentity,
    FrequencySamplePublicationManifest, NumericalPlanIdentity, SequenceIdentity, StepIdentity,
    StudyIdentity,
};
use runmat_analysis_fea::planning::resource::{
    build_acoustic_frequency_assembly_resource_plan, AcousticFrequencyAssemblyResourceDemand,
    ByteCount, Count, FeaResourcePlan,
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace,
    ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use std::collections::{BTreeMap, BTreeSet};

pub struct AcousticFrequencyAssemblyPreparation {
    pub workload: AcousticFrequencyWorkload,
    pub mesh: SolverMeshArtifact,
    pub entries: Vec<PreparedAnalysisObjects>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
}

pub struct PreparedAcousticFrequencyAssemblyTask {
    pub output_policy: AnalysisOutputPolicy,
    pub workload: AcousticFrequencyWorkload,
    pub mesh: SolverMeshArtifact,
    pub entries: Vec<AcousticHarmonicSolverResult>,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
}

pub struct ImportedAcousticFrequencyAssembly {
    pub result: AcousticFrequencyResult,
    pub publication: FrequencySamplePublicationManifest,
    pub result_objects: PreparedAnalysisObjects,
    pub publication_objects: PreparedAnalysisObjects,
    pub fields: Vec<PreparedAnalysisObjects>,
}

pub fn validate_acoustic_frequency_assembly_entries(
    workload: &AcousticFrequencyWorkload,
    mesh: &SolverMeshArtifact,
    entries: &[AcousticHarmonicSolverResult],
) -> AnalysisBridgeResult<()> {
    workload.validate()?;
    mesh.validate_canonical().map_err(invalid)?;
    let plan = workload.plan().entries().map_err(invalid)?;
    if entries.len() != plan.len() {
        return Err(invalid("acoustic assembly requires the complete plan"));
    }
    let mut by_frequency = BTreeMap::new();
    for result in entries {
        result.validate()?;
        if by_frequency
            .insert(result.frequency().hertz.to_bits(), result)
            .is_some()
        {
            return Err(invalid("duplicate acoustic assembly entry"));
        }
    }
    for entry in plan {
        let result = by_frequency
            .remove(&entry.frequency_hz().to_bits())
            .ok_or_else(|| invalid("missing acoustic assembly entry"))?;
        result.validate_for(&workload.entry(entry)?, mesh)?;
    }
    if !by_frequency.is_empty() {
        return Err(invalid("foreign acoustic assembly entry"));
    }
    Ok(())
}

pub fn decode_acoustic_frequency_assembly_entries(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<Vec<AcousticHarmonicSolverResult>> {
    inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::SolverResult)
        .map(|input| {
            let payload = input.payload();
            let result = AcousticHarmonicSolverResult::from_canonical_bytes(&payload)?;
            if input.manifest.logical_identity != result.source_result_identity()?.digest()
                || input.manifest.payload_digest != Digest::sha256(&payload)
                || input.manifest.encoded_length != payload.len() as u64
                || input.root.digest != Digest::sha256(input.manifest.canonical_bytes()?)
            {
                return Err(AnalysisBridgeError::Identity(
                    "acoustic entry object differs from original result",
                ));
            }
            Ok(result)
        })
        .collect()
}

/// Derive the complete envelope without producing an aggregate or recovering fields on the driver.
pub fn rederive_acoustic_frequency_assembly_resource_plan(
    workload: &AcousticFrequencyWorkload,
    mesh: &SolverMeshArtifact,
    entries: &[AcousticHarmonicSolverResult],
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<FeaResourcePlan> {
    validate_acoustic_frequency_assembly_entries(workload, mesh, entries)?;
    let numerical = inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .cloned()
        .collect::<Vec<_>>();
    let facts = crate::prepared_input_resource_facts(&numerical)?;
    let count = u64::try_from(entries.len()).map_err(invalid)?;
    // Each entry retains native complex pressure/velocity. Field partition/catalog overhead is
    // bounded per entry and per original numerical value, not by representative-frequency size.
    let values = entries.iter().try_fold(0_u64, |total, result| {
        let nodes = u64::try_from(result.ordered_node_ids().len()).map_err(invalid)?;
        let elements = u64::try_from(result.ordered_element_ids().len()).map_err(invalid)?;
        nodes
            .checked_mul(2)
            .and_then(|n| elements.checked_mul(6).and_then(|e| n.checked_add(e)))
            .and_then(|n| total.checked_add(n))
            .ok_or_else(|| invalid("acoustic assembly value count overflow"))
    })?;
    let result_profile =
        AcousticFrequencyResult::assembly_encoding_profile(workload, mesh, entries)?;
    let result_output = profiled_output_facts(result_profile)?;
    let mut publication_bytes = count
        .checked_mul(64 * 1024)
        .ok_or_else(|| invalid("acoustic publication metadata overflow"))?;
    let mut field_profile = runmat_canonical_codec::SerializedCanonicalProfile {
        encoded_bytes_upper_bound: 0,
        materialization_resident_bytes_estimate: 0,
    };
    for entry in entries {
        let (catalog, fields) = entry.native_field_encoding_profiles()?;
        // Authority, plan, recovery request and fixed native definitions fit the per-entry
        // allowance. Every growing partition descriptor is covered by its catalog profile.
        // The metadata contains no pressure, velocity, mesh or original solver arrays.
        publication_bytes = publication_bytes
            .checked_add(catalog.encoded_bytes_upper_bound)
            .ok_or_else(|| invalid("acoustic publication catalog overflow"))?;
        field_profile.encoded_bytes_upper_bound = field_profile
            .encoded_bytes_upper_bound
            .checked_add(fields.encoded_bytes_upper_bound)
            .ok_or_else(|| invalid("acoustic field payload profile overflow"))?;
        field_profile.materialization_resident_bytes_estimate = field_profile
            .materialization_resident_bytes_estimate
            .checked_add(fields.materialization_resident_bytes_estimate)
            .ok_or_else(|| invalid("acoustic field resident profile overflow"))?;
    }
    let publication_output = crate::projected_output_resource_facts(publication_bytes)?;
    // Native fields have one independent analysis manifest per frequency. A single
    // aggregate payload allowance must not silently account for just one field root.
    let mut fields_output = profiled_output_facts(field_profile)?;
    let extra_roots = count
        .checked_sub(1)
        .ok_or_else(|| invalid("empty acoustic field output"))?;
    let root_allowance = crate::projected_output_resource_facts(1)?;
    fields_output.transport_closure_bytes_upper_bound.0 = fields_output
        .transport_closure_bytes_upper_bound
        .0
        .checked_add(
            root_allowance
                .transport_closure_bytes_upper_bound
                .0
                .checked_mul(extra_roots)
                .ok_or_else(|| invalid("acoustic field closure overflow"))?,
        )
        .ok_or_else(|| invalid("acoustic field closure overflow"))?;
    fields_output.construction_resident_bytes_admission_budget.0 = fields_output
        .construction_resident_bytes_admission_budget
        .0
        .checked_add(
            root_allowance
                .construction_resident_bytes_admission_budget
                .0
                .checked_mul(extra_roots)
                .ok_or_else(|| invalid("acoustic field construction overflow"))?,
        )
        .ok_or_else(|| invalid("acoustic field construction overflow"))?;
    let workspace = facts
        .decoded_resident_bytes_admission_budget
        .0
        .checked_add(
            values
                .checked_mul(64)
                .ok_or_else(|| invalid("acoustic assembly workspace overflow"))?,
        )
        .ok_or_else(|| invalid("acoustic assembly workspace overflow"))?;
    let reference = workload.plan().entries().map_err(invalid)?[0];
    let harmonic = workload.entry(reference)?;
    let authored = harmonic
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == harmonic.step_id)
        .ok_or_else(|| invalid("acoustic assembly authored producer absent"))?
        .accuracy
        .resources;
    Ok(build_acoustic_frequency_assembly_resource_plan(
        AcousticFrequencyAssemblyResourceDemand {
            workload: workload.identity()?,
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            entry_count: Count(count),
            inputs: facts,
            workspace_bytes_upper_bound: ByteCount(workspace),
            result_output,
            publication_output,
            fields_output,
        },
        crate::projected_output_set_resource_facts(),
        authored,
    )?)
}

/// Same shared transport envelope, with the actual borrowed serializer's structural
/// materialization profile instead of a worst-density multiplier for numeric arrays.
fn profiled_output_facts(
    profile: runmat_canonical_codec::SerializedCanonicalProfile,
) -> AnalysisBridgeResult<runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts> {
    let mut output = crate::projected_output_resource_facts(profile.encoded_bytes_upper_bound)?;
    output.construction_resident_bytes_admission_budget = ByteCount(
        profile
            .materialization_resident_bytes_estimate
            .checked_add(output.transport_closure_bytes_upper_bound.0)
            .ok_or_else(|| invalid("profiled output construction overflow"))?,
    );
    Ok(output)
}

pub fn prepare_acoustic_frequency_assembly_task(
    preparation: AcousticFrequencyAssemblyPreparation,
) -> AnalysisBridgeResult<PreparedAcousticFrequencyAssemblyTask> {
    let policy = preparation.output_policy;
    if preparation
        .entries
        .iter()
        .any(|input| input.manifest.kind != AnalysisNumericalObjectKind::SolverResult)
    {
        return Err(invalid(
            "acoustic assembly accepts only original result entry objects",
        ));
    }
    let mut inputs = vec![
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::AcousticFrequencyWorkload,
            preparation.workload.identity()?.digest(),
            &preparation.workload.canonical_bytes()?,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?,
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::Mesh,
            Digest::from_bytes(*preparation.mesh.canonical_digest.bytes()),
            &preparation.mesh.canonical_encode().map_err(invalid)?,
            policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            policy.inventory_limits(),
        )?,
    ];
    inputs.extend(preparation.entries);
    let entries = decode_acoustic_frequency_assembly_entries(&inputs)?;
    let resource_plan = rederive_acoustic_frequency_assembly_resource_plan(
        &preparation.workload,
        &preparation.mesh,
        &entries,
        &inputs,
    )?;
    inputs.push(prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        policy.inventory_limits(),
    )?);
    let first = preparation.workload.plan().entries().map_err(invalid)?[0];
    let harmonic = preparation.workload.entry(first)?;
    let sequence = SequenceIdentity::from_solver_contract(&harmonic.model.sequence)?;
    let step = harmonic
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == harmonic.step_id)
        .ok_or_else(|| invalid("acoustic assembly producer absent"))?;
    let execution = ExecutionPlanIdentity::new(
        NumericalPlanIdentity::from_acoustic_frequency_workload(
            preparation.workload.identity()?,
            Digest::from_bytes(*preparation.mesh.canonical_digest.bytes()),
        )?,
        resource_plan.identity.digest(),
    )?;
    let mut outputs = vec![
        AnalysisNumericalObjectKind::SolverResult,
        AnalysisNumericalObjectKind::FrequencySamplePublication,
    ];
    outputs.extend(std::iter::repeat_n(
        AnalysisNumericalObjectKind::Field,
        entries.len(),
    ));
    let request = AnalysisWorkloadRequest::new(
        StudyIdentity::from_authored_digest(
            preparation.authored_study_digest,
            preparation.frozen_project_digest,
        )?,
        StepIdentity::from_solver_contract(sequence, step)?,
        execution,
        AnalysisWorkKind::AssembleAcousticFrequencyResult,
        inputs
            .iter()
            .map(|input| AnalysisWorkloadInput {
                kind: input.manifest.kind,
                logical_identity: input.manifest.logical_identity,
                payload_digest: input.manifest.payload_digest,
                manifest_digest: input.root.digest,
                manifest_encoded_length: input.root.encoded_length,
                reference_kind: if input.manifest.kind == AnalysisNumericalObjectKind::SolverResult
                {
                    ValueRefKind::ResultObject
                } else {
                    ValueRefKind::DriverObject
                },
            })
            .collect(),
        outputs,
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(request, preparation.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedAcousticFrequencyAssemblyTask {
        output_policy: policy,
        workload: preparation.workload,
        mesh: preparation.mesh,
        entries,
        host,
        program,
        inputs,
        resource_plan,
    })
}

pub fn import_acoustic_frequency_assembly(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedAcousticFrequencyAssemblyTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedAcousticFrequencyAssembly> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    if output_set.manifest.outputs.len() != prepared.entries.len() + 2 {
        return Err(invalid("acoustic assembly output count differs"));
    }
    let mut imported = Vec::new();
    for output in &output_set.manifest.outputs {
        imported.push(import_analysis_objects(
            source,
            &output.root,
            output.kind,
            output.logical_identity,
            output.payload_digest,
            &prepared.host.artifact_access,
            limits,
        )?);
    }
    let unique = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = imported.iter().filter(|input| input.manifest.kind == kind);
        let value = matches
            .next()
            .ok_or_else(|| invalid("acoustic assembly output omitted"))?;
        if matches.next().is_some() {
            return Err(invalid("duplicate acoustic assembly metadata output"));
        }
        Ok(value)
    };
    let result_objects = unique(AnalysisNumericalObjectKind::SolverResult)?.clone();
    let publication_objects =
        unique(AnalysisNumericalObjectKind::FrequencySamplePublication)?.clone();
    let result = AcousticFrequencyResult::from_canonical_bytes(&result_objects.payload())?;
    result.validate_for(&prepared.workload, &prepared.mesh)?;
    let expected = runmat_analysis_fea::execution::assemble_acoustic_frequency_result(
        prepared.workload.clone(),
        &prepared.mesh,
        prepared.entries.clone(),
    )?;
    let publication =
        FrequencySamplePublicationManifest::from_canonical_bytes(&publication_objects.payload())?;
    if result != expected.result
        || publication != expected.publication
        || result_objects.manifest.logical_identity != result.source_result_identity()?.digest()
        || publication_objects.manifest.logical_identity != publication.identity().digest()
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic assembly output differs from original entries",
        ));
    }
    let mut by_identity = BTreeMap::new();
    for objects in imported
        .iter()
        .filter(|objects| objects.manifest.kind == AnalysisNumericalObjectKind::Field)
    {
        if by_identity
            .insert(objects.manifest.logical_identity, objects)
            .is_some()
        {
            return Err(invalid("duplicate acoustic field output"));
        }
    }
    let mut fields = Vec::new();
    for artifact in &expected.fields {
        let objects = by_identity
            .remove(&artifact.identity().digest())
            .ok_or_else(|| invalid("missing original acoustic field output"))?;
        let expected_partitions =
            crate::field_recovery::recovered_field_payload_partitions(artifact)?;
        if objects.manifest.partitions.len() != expected_partitions.len() {
            return Err(AnalysisBridgeError::Identity(
                "acoustic field partitions differ from original native fields",
            ));
        }
        for (declared, (identity, payload)) in
            objects.manifest.partitions.iter().zip(expected_partitions)
        {
            if declared.logical_identity != identity
                || declared.encoded_length != payload.len() as u64
                || declared.payload_digest != Digest::sha256(&payload)
                || objects.partition_payload(identity)? != payload
            {
                return Err(AnalysisBridgeError::Identity(
                    "acoustic field partitions differ from original native fields",
                ));
            }
        }
        fields.push(objects.clone());
    }
    if !by_identity.is_empty() {
        return Err(invalid("foreign acoustic field output"));
    }
    let mut closure = Vec::new();
    for objects in &imported {
        closure.extend(
            objects
                .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
        );
    }
    crate::object::retain_unique_references(&mut closure);
    closure.push(output_set.root);
    if closure != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "acoustic assembly inventory differs from canonical closure",
        ));
    }
    Ok(ImportedAcousticFrequencyAssembly {
        result,
        publication,
        result_objects,
        publication_objects,
        fields,
    })
}

pub fn prepare_acoustic_frequency_field_objects(
    artifact: &runmat_analysis_fea::results::field::RecoveredFieldArtifact,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    crate::prepare_recovered_field_objects(
        artifact,
        chunk_bytes,
        ObjectNamespace::ResultValue,
        limits,
    )
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
