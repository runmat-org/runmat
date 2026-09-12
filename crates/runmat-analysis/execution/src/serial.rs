use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisPayloadPartition, AnalysisResultIdentity,
    AnalysisWorkKind, DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionRequest};

use crate::{
    import_analysis_objects, prepare_analysis_objects, prepare_partitioned_analysis_objects,
    response::{publish_analysis_output_set, AnalysisOutputSet},
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisKernelOutput {
    pub kind: AnalysisNumericalObjectKind,
    pub logical_identity: Digest,
    pub payload: Vec<u8>,
    pub partitions: Vec<AnalysisKernelOutputPartition>,
}

impl AnalysisKernelOutput {
    pub fn whole(
        kind: AnalysisNumericalObjectKind,
        logical_identity: Digest,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            kind,
            logical_identity,
            payload,
            partitions: Vec::new(),
        }
    }

    pub fn partitioned(
        kind: AnalysisNumericalObjectKind,
        logical_identity: Digest,
        partitions: Vec<AnalysisKernelOutputPartition>,
    ) -> Self {
        Self {
            kind,
            logical_identity,
            payload: Vec::new(),
            partitions,
        }
    }

    fn is_valid(&self) -> bool {
        (self.partitions.is_empty() && !self.payload.is_empty())
            || (self.payload.is_empty()
                && !self.partitions.is_empty()
                && self
                    .partitions
                    .iter()
                    .all(|partition| !partition.payload.is_empty()))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisKernelOutputPartition {
    pub logical_identity: Digest,
    pub payload: Vec<u8>,
}

pub struct AnalysisKernelInvocation<'a> {
    pub kind: AnalysisWorkKind,
    pub host: &'a AnalysisHostWorkload,
    pub inputs: &'a [PreparedAnalysisObjects],
}

/// Executes analysis-owned numerical semantics after the bridge has closed and validated every
/// input artifact. Implementations must not retain borrowed payloads or publish objects directly.
pub trait AnalysisKernel: Send + Sync {
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletedAnalysisWorkload {
    pub identity: AnalysisResultIdentity,
    pub outputs: Vec<PreparedAnalysisObjects>,
    pub roots: Vec<ValueRef>,
    pub output_set: AnalysisOutputSet,
    pub output_set_root: ValueRef,
    pub result_objects: Vec<ValueRef>,
}

impl CompletedAnalysisWorkload {
    pub fn attempt_success(&self) -> runmat_execution_runner::AttemptSuccess {
        runmat_execution_runner::AttemptSuccess {
            outputs: vec![ValuePayload::Object(Box::new(self.output_set_root.clone()))],
            result_objects: self.result_objects.clone(),
            resources: None,
        }
    }
}

pub fn execute_serial_analysis<S, K>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &K,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<CompletedAnalysisWorkload>
where
    S: CacheImport + CacheExport,
    K: AnalysisKernel + ?Sized,
{
    if chunk_bytes == 0
        || limits.max_objects == 0
        || limits.max_object_bytes == 0
        || limits.max_total_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "analysis execution limits must be non-zero".into(),
        ));
    }
    let host = AnalysisHostWorkload::from_program_request(request, store)?;
    let roots = input_roots(request)?;
    let inputs = host
        .workload
        .inputs
        .iter()
        .zip(&roots)
        .map(|(input, root)| {
            import_analysis_objects(
                store,
                root,
                input.kind,
                input.logical_identity,
                input.payload_digest,
                &host.artifact_access,
                limits,
            )
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let kernel_outputs = kernel.execute(AnalysisKernelInvocation {
        kind: host.workload.kind,
        host: &host,
        inputs: &inputs,
    })?;
    if kernel_outputs.len() != host.workload.outputs.len()
        || kernel_outputs
            .iter()
            .zip(&host.workload.outputs)
            .any(|(actual, expected)| actual.kind != *expected || !actual.is_valid())
    {
        return Err(AnalysisBridgeError::Invalid(
            "analysis kernel outputs differ from the workload contract".into(),
        ));
    }

    let mut outputs = Vec::with_capacity(kernel_outputs.len());
    let mut roots = Vec::with_capacity(kernel_outputs.len());
    let mut result_objects = Vec::new();
    for output in kernel_outputs {
        let bounded_chunk_bytes = chunk_bytes.min(DEFAULT_ANALYSIS_CHUNK_BYTES);
        let prepared = if output.partitions.is_empty() {
            prepare_analysis_objects(
                output.kind,
                output.logical_identity,
                &output.payload,
                bounded_chunk_bytes,
                ObjectNamespace::ResultValue,
                limits,
            )?
        } else {
            let partitions = output
                .partitions
                .iter()
                .map(|partition| AnalysisPayloadPartition {
                    logical_identity: partition.logical_identity,
                    payload: &partition.payload,
                })
                .collect::<Vec<_>>();
            prepare_partitioned_analysis_objects(
                output.kind,
                output.logical_identity,
                &partitions,
                bounded_chunk_bytes,
                ObjectNamespace::ResultValue,
                limits,
            )?
        };
        let references =
            prepared.unique_references(&host.artifact_access, ValueRefKind::ResultObject)?;
        let root = references
            .first()
            .cloned()
            .ok_or(AnalysisBridgeError::Identity(
                "analysis output closure has no manifest root",
            ))?;
        for object in &prepared.objects {
            store.write_verified(object)?;
        }
        roots.push(root);
        result_objects.extend(references);
        outputs.push(prepared);
    }
    let identity = AnalysisResultIdentity::new(
        host.workload.identity,
        outputs
            .iter()
            .map(|output| output.manifest.payload_digest)
            .collect(),
    )?;
    let (output_set, output_set_root) =
        publish_analysis_output_set(store, &host, identity, &outputs, &roots)?;
    crate::object::retain_unique_references(&mut result_objects);
    result_objects.push(output_set_root.clone());
    Ok(CompletedAnalysisWorkload {
        identity,
        outputs,
        roots,
        output_set,
        output_set_root,
        result_objects,
    })
}

fn input_roots(request: &ProgramExecutionRequest) -> AnalysisBridgeResult<Vec<ValueRef>> {
    AnalysisHostWorkload::semantic_input_roots(request)
}
