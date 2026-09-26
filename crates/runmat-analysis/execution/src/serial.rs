use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisResultIdentity, AnalysisWorkKind,
};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionRequest;

use crate::{
    import_analysis_objects,
    response::{prepare_analysis_output_set, AnalysisOutputSet},
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};

mod output;
pub use output::{
    AnalysisKernelOutput, AnalysisKernelOutputPartition, AnalysisKernelPayload,
    AnalysisOutputPolicy,
};

pub struct AnalysisKernelInvocation<'a> {
    pub output_policy: AnalysisOutputPolicy,
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
    let output_policy = AnalysisOutputPolicy::new(chunk_bytes, limits)?;
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
    let resource_plan_inputs = inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    if resource_plan_inputs.len() > 1 {
        return Err(AnalysisBridgeError::Invalid(
            "analysis execution has more than one resource plan".into(),
        ));
    }
    let resource_plan = resource_plan_inputs
        .first()
        .map(|input| {
            runmat_analysis_fea::planning::resource::FeaResourcePlan::from_canonical_bytes(
                &input.payload(),
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;

    let kernel_outputs = kernel.execute(AnalysisKernelInvocation {
        output_policy,
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
        let admitted_output = resource_plan
            .as_ref()
            .and_then(|plan| plan.input.demand.output_facts(output.kind));
        if resource_plan.as_ref().is_some_and(|plan| {
            plan.input.demand.governs_prepared_outputs() && admitted_output.is_none()
        }) {
            return Err(AnalysisBridgeError::Invalid(
                "analysis output has no envelope in its resource plan".into(),
            ));
        }
        let prepared = output.into_prepared(
            output_policy.chunk_bytes(),
            output_policy.inventory_limits(),
        )?;
        if let Some(admitted) = admitted_output {
            crate::object::validate_prepared_output_resource_facts(&prepared, admitted)?;
        }
        let references =
            prepared.unique_references(&host.artifact_access, ValueRefKind::ResultObject)?;
        let root = references
            .first()
            .cloned()
            .ok_or(AnalysisBridgeError::Identity(
                "analysis output closure has no manifest root",
            ))?;
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
    let prepared_output_set = prepare_analysis_output_set(&host, identity, &outputs, &roots)?;
    if let Some(plan) = &resource_plan {
        if plan.input.demand.governs_prepared_outputs() {
            let admitted = plan.input.output_set.as_ref().ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "analysis output set has no envelope in its resource plan".into(),
                )
            })?;
            prepared_output_set.validate_resource_facts(admitted)?;
        }
    }
    for output in &outputs {
        for object in &output.objects {
            store.write_verified(object)?;
        }
    }
    store.write_verified(&prepared_output_set.object)?;
    let output_set = prepared_output_set.output_set;
    let output_set_root = prepared_output_set.root;
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
