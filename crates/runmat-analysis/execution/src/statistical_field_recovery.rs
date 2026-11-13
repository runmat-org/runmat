//! Source-bound recovery of a statistical step's fields (random-response RMS, response-spectrum
//! combined peak) for authored observations. The task copies the original assembled result's
//! fields; it never re-solves.
use crate::*;
use runmat_analysis_fea::{
    execution::{
        random_response_statistical_fields, response_spectrum_statistical_fields,
        statistical_field_request, statistical_recovered_fields, AnalysisNumericalObjectKind,
        AnalysisTransportProfiler, AnalysisWorkKind, ExecutionPlanIdentity, FieldRecoveryRequest,
        NumericalPlanIdentity, SolverResultIdentity, StatisticalFieldValues, StatisticalProcedure,
        StepIdentity, StructuralModalResponseSpectrumSolverResult,
        StructuralRandomResponseSolverResult,
    },
    planning::resource::{
        build_statistical_field_recovery_resource_plan, statistical_field_recovery_shape,
        ByteCount, FeaResourcePlan, PreparedPhaseResourceFacts,
    },
    results::field::RecoveredFieldArtifact,
    solve::dynamics::ResponseSpectrumFieldRecovery,
};
use runmat_execution::{value::ValueRefKind, Digest, ProgramRevision};
use runmat_execution_artifact::ObjectNamespace;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct StatisticalFieldRecoveryPreparation {
    /// The completed random-response or response-spectrum assembly.
    pub source_host: AnalysisHostWorkload,
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    /// The response-spectrum assembly's own field recovery; absent for random response.
    pub source_fields: Option<PreparedAnalysisObjects>,
    pub program_revision: ProgramRevision,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub struct PreparedStatisticalFieldRecovery {
    pub procedure: StatisticalProcedure,
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: FeaResourcePlan,
    pub output_policy: AnalysisOutputPolicy,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

/// The statistical procedure of a recovery's immutable inputs, from its workload kind.
pub fn statistical_procedure(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<StatisticalProcedure> {
    let kinds = inputs
        .iter()
        .map(|input| input.manifest.kind)
        .collect::<Vec<_>>();
    match (
        kinds.contains(&AnalysisNumericalObjectKind::StructuralRandomResponseWorkload),
        kinds.contains(&AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload),
    ) {
        (true, false) => Ok(StatisticalProcedure::RandomResponse),
        (false, true) => Ok(StatisticalProcedure::ResponseSpectrum),
        _ => Err(invalid(
            "statistical recovery requires exactly one statistical workload",
        )),
    }
}

fn workload_kind(procedure: StatisticalProcedure) -> AnalysisNumericalObjectKind {
    match procedure {
        StatisticalProcedure::RandomResponse => {
            AnalysisNumericalObjectKind::StructuralRandomResponseWorkload
        }
        StatisticalProcedure::ResponseSpectrum => {
            AnalysisNumericalObjectKind::StructuralModalResponseSpectrumWorkload
        }
    }
}

fn input_kinds(procedure: StatisticalProcedure) -> Vec<AnalysisNumericalObjectKind> {
    let mut kinds = vec![
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        workload_kind(procedure),
        AnalysisNumericalObjectKind::Mesh,
        AnalysisNumericalObjectKind::SolverResult,
    ];
    if procedure == StatisticalProcedure::ResponseSpectrum {
        kinds.push(AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery);
    }
    kinds
}

pub fn prepare_statistical_field_recovery(
    p: StatisticalFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedStatisticalFieldRecovery> {
    p.source_host.validate()?;
    let procedure = match p.source_host.workload.kind {
        AnalysisWorkKind::AssembleStructuralRandomResponseResult => {
            StatisticalProcedure::RandomResponse
        }
        AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult => {
            StatisticalProcedure::ResponseSpectrum
        }
        _ => {
            return Err(invalid(
                "statistical field recovery requires its completed assembly",
            ))
        }
    };
    if p.source_fields.is_some() != (procedure == StatisticalProcedure::ResponseSpectrum) {
        return Err(invalid(
            "statistical field recovery source inventory differs from its procedure",
        ));
    }
    for (kind, objects) in [
        (workload_kind(procedure), &p.workload),
        (AnalysisNumericalObjectKind::Mesh, &p.mesh),
    ] {
        let source = p
            .source_host
            .workload
            .inputs
            .iter()
            .filter(|input| input.kind == kind)
            .collect::<Vec<_>>();
        if source.len() != 1
            || objects.manifest.kind != kind
            || source[0].logical_identity != objects.manifest.logical_identity
            || source[0].payload_digest != objects.manifest.payload_digest
            || source[0].manifest_digest != objects.root.digest
            || source[0].manifest_encoded_length != objects.root.encoded_length
        {
            return Err(invalid(
                "statistical recovery original source closure differs",
            ));
        }
    }
    let source_identity = SolverResultIdentity::new(
        p.source_host.workload.identity,
        p.source_result.manifest.payload_digest,
    )?;
    let mut inputs = vec![p.workload, p.mesh, p.source_result];
    inputs.extend(p.source_fields);
    let request =
        statistical_recovery_request(p.source_host.workload.step, source_identity, &inputs)?;
    inputs.insert(
        0,
        prepare_analysis_objects(
            AnalysisNumericalObjectKind::FieldRecoveryRequest,
            request.identity().digest(),
            &request.canonical_bytes()?,
            p.output_policy.chunk_bytes(),
            ObjectNamespace::InputValue,
            p.output_policy.inventory_limits(),
        )?,
    );
    let resource_plan = build_prepared_statistical_field_recovery_resource_plan(
        &request,
        &inputs,
        p.output_policy,
        p.local_execution,
    )?;
    inputs.push(prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        resource_plan.identity.digest(),
        &resource_plan.canonical_bytes()?,
        p.output_policy.chunk_bytes(),
        ObjectNamespace::InputValue,
        p.output_policy.inventory_limits(),
    )?);
    let execution = execution_plan(&request, resource_plan.identity.digest())?;
    let request_host = AnalysisWorkloadRequest::new(
        p.source_host.workload.study,
        p.source_host.workload.step,
        execution,
        AnalysisWorkKind::RecoverField,
        host_inputs(&inputs),
        vec![AnalysisNumericalObjectKind::Field],
        p.source_host.workload.required_capabilities.clone(),
    )?;
    let host =
        AnalysisHostWorkload::new(request_host, p.source_host.artifact_access, p.output_policy)?;
    let program = prepare_analysis_program(&host, p.program_revision, &inputs)?;
    Ok(PreparedStatisticalFieldRecovery {
        procedure,
        request,
        host,
        program,
        inputs,
        resource_plan,
        output_policy: p.output_policy,
        local_execution: p.local_execution,
    })
}

/// Decoded, cross-validated statistical sources.
pub struct DecodedStatisticalSource {
    pub mesh: SolverMeshArtifact,
    source: StatisticalValues,
}

enum StatisticalValues {
    Random(Box<StructuralRandomResponseSolverResult>),
    Spectrum(Box<ResponseSpectrumFieldRecovery>),
}

impl DecodedStatisticalSource {
    pub fn fields(&self) -> Vec<StatisticalFieldValues<'_>> {
        match &self.source {
            StatisticalValues::Random(result) => {
                random_response_statistical_fields(result.numerical())
            }
            StatisticalValues::Spectrum(recovery) => response_spectrum_statistical_fields(recovery),
        }
    }

    pub fn recover(
        &self,
        request: &FieldRecoveryRequest,
    ) -> AnalysisBridgeResult<RecoveredFieldArtifact> {
        Ok(statistical_recovered_fields(
            request,
            &self.mesh,
            &self.fields(),
        )?)
    }
}

fn unique(
    inputs: &[PreparedAnalysisObjects],
    kind: AnalysisNumericalObjectKind,
) -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
    let mut matching = inputs.iter().filter(|input| input.manifest.kind == kind);
    match (matching.next(), matching.next()) {
        (Some(input), None) => Ok(input),
        _ => Err(invalid("statistical recovery input is absent or repeated")),
    }
}

/// Decode and cross-validate the immutable recovery sources among `inputs`.
pub fn decode_statistical_recovery_source(
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<DecodedStatisticalSource> {
    let procedure = statistical_procedure(inputs)?;
    let mesh_objects = unique(inputs, AnalysisNumericalObjectKind::Mesh)?;
    let source = unique(inputs, AnalysisNumericalObjectKind::SolverResult)?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload()).map_err(invalid)?;
    if mesh_objects.manifest.logical_identity != Digest::from_bytes(*mesh.canonical_digest.bytes())
    {
        return Err(invalid("statistical recovery mesh identity differs"));
    }
    let values = match procedure {
        StatisticalProcedure::RandomResponse => StatisticalValues::Random(Box::new(
            StructuralRandomResponseSolverResult::from_canonical_bytes(&source.payload())?,
        )),
        StatisticalProcedure::ResponseSpectrum => {
            let result = StructuralModalResponseSpectrumSolverResult::from_canonical_bytes(
                &source.payload(),
            )?;
            let recovery = ResponseSpectrumFieldRecovery::from_canonical_bytes(
                &unique(
                    inputs,
                    AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery,
                )?
                .payload(),
            )
            .map_err(invalid)?;
            recovery
                .validate_for_result(result.numerical())
                .map_err(invalid)?;
            StatisticalValues::Spectrum(Box::new(recovery))
        }
    };
    Ok(DecodedStatisticalSource {
        mesh,
        source: values,
    })
}

/// The one recovery request of an exact statistical result on its canonical mesh.
pub fn statistical_recovery_request(
    producer: StepIdentity,
    identity: SolverResultIdentity,
    inputs: &[PreparedAnalysisObjects],
) -> AnalysisBridgeResult<FieldRecoveryRequest> {
    let source = unique(inputs, AnalysisNumericalObjectKind::SolverResult)?;
    if identity.digest() != source.manifest.logical_identity {
        return Err(invalid("statistical recovery source identity differs"));
    }
    let decoded = decode_statistical_recovery_source(inputs)?;
    statistical_field_request(producer, identity, &decoded.mesh, &decoded.fields())
        .map_err(Into::into)
}

fn execution_plan(
    request: &FieldRecoveryRequest,
    resource_plan: Digest,
) -> AnalysisBridgeResult<ExecutionPlanIdentity> {
    let source = request
        .source
        .solver_result()
        .ok_or_else(|| invalid("statistical recovery requires its solver result"))?;
    let numerical = NumericalPlanIdentity::from_statistical_field_recovery(
        source,
        request.topology.discretization_digest(),
    )?;
    Ok(ExecutionPlanIdentity::new(numerical, resource_plan)?)
}

/// The execution plan a statistical recovery host must carry.
pub fn statistical_recovery_execution_plan(
    request: &FieldRecoveryRequest,
    resource_plan: Digest,
) -> AnalysisBridgeResult<ExecutionPlanIdentity> {
    execution_plan(request, resource_plan)
}

fn host_inputs(inputs: &[PreparedAnalysisObjects]) -> Vec<AnalysisWorkloadInput> {
    inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: if matches!(
                input.manifest.kind,
                AnalysisNumericalObjectKind::SolverResult
                    | AnalysisNumericalObjectKind::StructuralModalResponseSpectrumFieldRecovery
            ) {
                ValueRefKind::ResultObject
            } else {
                ValueRefKind::DriverObject
            },
        })
        .collect()
}

/// Metadata-only rederivation used by preparation, the worker and import alike.
pub fn build_prepared_statistical_field_recovery_resource_plan(
    request: &FieldRecoveryRequest,
    inputs: &[PreparedAnalysisObjects],
    policy: AnalysisOutputPolicy,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> AnalysisBridgeResult<FeaResourcePlan> {
    let procedure = statistical_procedure(inputs)?;
    let numerical = inputs
        .iter()
        .filter(|i| i.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    let kinds = input_kinds(procedure);
    if numerical.len() != kinds.len()
        || inputs.len() - numerical.len() > 1
        || kinds.iter().any(|kind| {
            numerical
                .iter()
                .filter(|i| i.manifest.kind == *kind)
                .count()
                != 1
        })
    {
        return Err(invalid(
            "statistical recovery immutable input inventory differs",
        ));
    }
    if unique(inputs, AnalysisNumericalObjectKind::FieldRecoveryRequest)?
        .manifest
        .logical_identity
        != request.identity().digest()
    {
        return Err(invalid("statistical recovery request differs"));
    }
    let (partitions, _) = statistical_field_recovery_shape(procedure, request)?;
    let mut profiler = AnalysisTransportProfiler::new(policy.chunk_bytes())?;
    for bytes in partitions {
        profiler.push(bytes)?;
    }
    let profile = profiler.finish(AnalysisNumericalObjectKind::Field)?;
    crate::thermal_execution::require_inventory(&profile, policy.inventory_limits())?;
    // Decoding the sources and validating their shapes is the only source work.
    let source_validation = numerical
        .iter()
        .map(|input| input.manifest.encoded_length)
        .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes))
        .ok_or_else(|| invalid("statistical recovery source bytes overflow"))?;
    Ok(build_statistical_field_recovery_resource_plan(
        procedure,
        request,
        ByteCount(source_validation.max(1)),
        PreparedPhaseResourceFacts {
            inputs: prepared_input_resource_facts_without_resource_plan(inputs)?,
            output: projected_transport_output_resource_facts(&profile)?,
            output_set: projected_output_set_resource_facts(),
        },
        local_execution,
    )?)
}

pub fn import_statistical_recovered_fields(
    source: &impl runmat_execution_artifact::cache::CacheImport,
    response: runmat_execution_artifact::ProgramExecutionResponse,
    prepared: &PreparedStatisticalFieldRecovery,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::RecoverField
        || prepared.host.workload.outputs != [AnalysisNumericalObjectKind::Field]
        || host_inputs(&prepared.inputs) != prepared.host.workload.inputs
    {
        return Err(invalid(
            "statistical field import requires its exact recovery task",
        ));
    }
    let expected = build_prepared_statistical_field_recovery_resource_plan(
        &prepared.request,
        &prepared.inputs,
        prepared.output_policy,
        prepared.local_execution,
    )?;
    if expected != prepared.resource_plan
        || prepared.host.workload.execution_plan
            != execution_plan(&prepared.request, expected.identity.digest())?
    {
        return Err(invalid(
            "statistical recovery import resource authority differs",
        ));
    }
    crate::field_recovery::import_recovered_fields(
        source,
        response,
        &prepared.request,
        &prepared.host,
        &prepared.program,
        limits,
    )
}

fn invalid(error: impl std::fmt::Display) -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(error.to_string())
}
