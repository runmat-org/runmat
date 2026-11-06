use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisResultIdentity, AnalysisWorkKind,
};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionRequest;

use crate::{
    response::{prepare_analysis_output_set, AnalysisOutputSet},
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
};

pub type AnalysisKernelFuture<'a> = std::pin::Pin<
    Box<dyn std::future::Future<Output = AnalysisBridgeResult<Vec<AnalysisKernelOutput>>> + 'a>,
>;

pub use runmat_simulation_usage::AsyncCapacityOwner;

mod output;
mod reservation;
pub use output::{
    AnalysisKernelOutput, AnalysisKernelOutputPartition, AnalysisKernelPayload,
    AnalysisOutputPolicy,
};
pub use reservation::ReservedAnalysisKernelInvocation;

pub struct AnalysisKernelInvocation<'a> {
    pub output_policy: AnalysisOutputPolicy,
    pub kind: AnalysisWorkKind,
    pub host: &'a AnalysisHostWorkload,
    pub inputs: &'a [PreparedAnalysisObjects],
}

/// Executes analysis-owned numerical semantics after the bridge has closed and validated every
/// input artifact. Implementations must not retain borrowed payloads or publish objects directly.
pub trait AnalysisKernel: Send + Sync {
    /// The shared boundary has compared the complete governing request with this reservation.
    /// Domain implementations may inspect execution capacity without changing physical identities.
    fn execute_reserved(
        &self,
        invocation: ReservedAnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>> {
        self.execute(invocation.into_invocation())
    }
    fn execute(
        &self,
        invocation: AnalysisKernelInvocation<'_>,
    ) -> AnalysisBridgeResult<Vec<AnalysisKernelOutput>>;

    fn execute_with_capacity<'a>(
        &'a self,
        invocation: AnalysisKernelInvocation<'a>,
        _capacity: &'a dyn AsyncCapacityOwner,
    ) -> AnalysisKernelFuture<'a> {
        Box::pin(async move { self.execute(invocation) })
    }

    fn execute_reserved_with_capacity<'a>(
        &'a self,
        invocation: ReservedAnalysisKernelInvocation<'a>,
        _capacity: &'a dyn AsyncCapacityOwner,
    ) -> AnalysisKernelFuture<'a> {
        Box::pin(async move { self.execute_reserved(invocation) })
    }
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
        runmat_execution_runner::AttemptSuccess::Values {
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
    execute_analysis(request, store, kernel, chunk_bytes, limits, None)
        .map(|(_, completed)| completed)
}

/// Scheduled execution receives the original scheduler reservation explicitly.
/// This does not infer an allocation from a numerical plan or a host's machine inventory.
pub fn execute_reserved_serial_analysis<S, K>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &K,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
    resources: &runmat_execution::resource::ResourceRequest,
) -> AnalysisBridgeResult<CompletedAnalysisWorkload>
where
    S: CacheImport + CacheExport,
    K: AnalysisKernel + ?Sized,
{
    resources
        .validate()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    execute_analysis(request, store, kernel, chunk_bytes, limits, Some(resources))
        .map(|(_, completed)| completed)
}

/// Construct the shared host response from the same admitted invocation used for execution.
/// `None` is explicit standalone execution; scheduled callers supply their carried reservation.
pub fn execute_serial_analysis_host_response<S, K>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &K,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
    resources: Option<&runmat_execution::resource::ResourceRequest>,
) -> AnalysisBridgeResult<crate::AnalysisHostResponse>
where
    S: CacheImport + CacheExport,
    K: AnalysisKernel + ?Sized,
{
    let (host, completed) =
        execute_analysis(request, store, kernel, chunk_bytes, limits, resources)?;
    crate::AnalysisHostResponse::completed(&host, &completed)
}

pub async fn execute_serial_analysis_host_response_with_capacity<S, K>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &K,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
    resources: Option<&runmat_execution::resource::ResourceRequest>,
    capacity: &dyn AsyncCapacityOwner,
) -> AnalysisBridgeResult<crate::AnalysisHostResponse>
where
    S: CacheImport + CacheExport,
    K: AnalysisKernel + ?Sized,
{
    let (host, completed) = execute_analysis_with_capacity(
        request,
        store,
        kernel,
        chunk_bytes,
        limits,
        resources,
        Some(capacity),
    )
    .await?;
    crate::AnalysisHostResponse::completed(&host, &completed)
}

pub(crate) fn finish_inline<T>(
    future: impl std::future::Future<Output = AnalysisBridgeResult<T>>,
) -> AnalysisBridgeResult<T> {
    use std::task::{Context, Poll, Waker};
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(result) => result,
        Poll::Pending => Err(AnalysisBridgeError::Identity(
            "synchronous analysis execution requires an immediately ready host",
        )),
    }
}

fn execute_analysis<S, K>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &K,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
    resources: Option<&runmat_execution::resource::ResourceRequest>,
) -> AnalysisBridgeResult<(AnalysisHostWorkload, CompletedAnalysisWorkload)>
where
    S: CacheImport + CacheExport,
    K: AnalysisKernel + ?Sized,
{
    finish_inline(execute_analysis_with_capacity(
        request,
        store,
        kernel,
        chunk_bytes,
        limits,
        resources,
        None,
    ))
}

async fn execute_analysis_with_capacity<S, K>(
    request: &ProgramExecutionRequest,
    store: &mut S,
    kernel: &K,
    chunk_bytes: usize,
    limits: ObjectInventoryLimits,
    resources: Option<&runmat_execution::resource::ResourceRequest>,
    capacity: Option<&dyn AsyncCapacityOwner>,
) -> AnalysisBridgeResult<(AnalysisHostWorkload, CompletedAnalysisWorkload)>
where
    S: CacheImport + CacheExport,
    K: AnalysisKernel + ?Sized,
{
    let admission = resources
        .map(|resources| {
            resources
                .validate()
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
            crate::object::AnalysisInputAdmission::new(request, resources.memory_bytes)
        })
        .transpose()?;
    let host =
        AnalysisHostWorkload::from_program_request_admitted(request, store, admission.as_ref())?;
    let output_policy = host.output_policy;
    // Inputs were admitted under the invocation's policy by their producers; outputs this
    // worker creates are bounded by its own reservation as well.
    let worker_policy =
        output_policy.bounded_by(AnalysisOutputPolicy::new(chunk_bytes, limits)?)?;
    let limits = output_policy.inventory_limits();
    let roots = input_roots(request)?;
    let inputs = host
        .workload
        .inputs
        .iter()
        .zip(&roots)
        .map(|(input, root)| {
            crate::object::import_analysis_objects_admitted(
                store,
                crate::object::AnalysisObjectImport {
                    root,
                    expected_kind: input.kind,
                    expected_logical_identity: input.logical_identity,
                    expected_payload_digest: input.payload_digest,
                    access: &host.artifact_access,
                    limits,
                },
                admission.as_ref(),
            )
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    let resource_plan_inputs = inputs
        .iter()
        .filter(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .collect::<Vec<_>>();
    if let Some(admission) = &admission {
        let facts =
            crate::prepared_input_resource_facts_from_refs(resource_plan_inputs.iter().copied())?;
        admission.require_additional(facts.decoded_resident_bytes_admission_budget.0)?;
    }
    let mut resource_plans = resource_plan_inputs
        .iter()
        .map(|input| crate::decode_admitted_resource_plan(input, limits))
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    // A publication may carry immutable upstream plans as original source authority.
    // Only one plan may govern this task's complete output inventory. Workers still
    // validate each source plan and the phase-specific exact immutable closure.
    let resource_plan = if resource_plans.len() <= 1 {
        resource_plans.pop()
    } else {
        let applicable = resource_plans
            .iter()
            .enumerate()
            .filter(|(_, plan)| {
                plan.input.demand.governs_prepared_outputs()
                    && host
                        .workload
                        .outputs
                        .iter()
                        .all(|kind| plan.input.demand.output_facts(*kind).is_some())
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let [index] = applicable.as_slice() else {
            return Err(AnalysisBridgeError::Identity(
                "analysis task requires exactly one output-governing resource plan",
            ));
        };
        Some(resource_plans.swap_remove(*index))
    };

    let invocation = AnalysisKernelInvocation {
        output_policy,
        kind: host.workload.kind,
        host: &host,
        inputs: &inputs,
    };
    let kernel_outputs = match resources {
        Some(resources) => {
            let invocation = ReservedAnalysisKernelInvocation::new(
                invocation,
                resources,
                resource_plan.as_ref(),
            )?;
            if let Some(admission) = &admission {
                admission.admit_payload_decoding(&inputs)?;
            }
            match capacity {
                Some(capacity) => {
                    kernel
                        .execute_reserved_with_capacity(invocation, capacity)
                        .await?
                }
                None => kernel.execute_reserved(invocation)?,
            }
        }
        None => match capacity {
            Some(capacity) => kernel.execute_with_capacity(invocation, capacity).await?,
            None => kernel.execute(invocation)?,
        },
    };
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
            worker_policy.chunk_bytes(),
            worker_policy.inventory_limits(),
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
    store.write_verified_batch(
        outputs
            .iter()
            .flat_map(|output| &output.objects)
            .chain(std::iter::once(&prepared_output_set.object)),
    )?;
    let output_set = prepared_output_set.output_set;
    let output_set_root = prepared_output_set.root;
    crate::object::retain_unique_references(&mut result_objects);
    result_objects.push(output_set_root.clone());
    Ok((
        host,
        CompletedAnalysisWorkload {
            identity,
            outputs,
            roots,
            output_set,
            output_set_root,
            result_objects,
        },
    ))
}

fn input_roots(request: &ProgramExecutionRequest) -> AnalysisBridgeResult<Vec<ValueRef>> {
    AnalysisHostWorkload::semantic_input_roots(request)
}
