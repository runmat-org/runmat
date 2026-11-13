use crate::fusion_exec::reduction::{CpuReadbackFailure, CpuReadbackPlan};
use crate::placement::LocalPlacementFeedback;
use runmat_accelerate_api::observation::ProviderOperationReceipt;
use runmat_execution::profiling::CommandProviderContribution;
use runmat_runtime::context::{RuntimeContext, VmFusionCompletionRight, VmFusionHostRoute};
use runmat_time::{duration_ns_saturating, Instant};
use runmat_value::Value;

mod host_preparation;
mod native_host;
mod native_original;
use host_preparation::{prepare_selected_host_inputs, prepare_vm_outputs};
use native_host::prepare_native_host;

/// Linear selected-provider receipt held until the VM has published all
/// ordered stores, adopted the result, and advanced beyond the fused span.
/// Dropping it on catch/deopt/cancellation never reports success or a fake
/// zero-duration failure.
pub struct FusionProviderCompletion {
    receipt: ProviderOperationReceipt,
    command_contribution: Option<CommandProviderContribution>,
    feedback: LocalPlacementFeedback,
    right: VmFusionCompletionRight,
}

impl FusionProviderCompletion {
    pub(super) fn new(
        receipt: ProviderOperationReceipt,
        feedback: LocalPlacementFeedback,
        right: VmFusionCompletionRight,
        command_contribution: Option<CommandProviderContribution>,
    ) -> Option<Self> {
        (feedback.is_current() && right.placement_revision() == Some(&feedback.signature.revision))
            .then_some(Self {
                receipt,
                command_contribution,
                feedback,
                right,
            })
    }

    pub fn complete(self, outer_started: Instant) {
        let Some(prelude) = self.receipt.default_clock_prelude_since(outer_started) else {
            return;
        };
        let Some(lease) = self.right.completion_lease(&self.feedback.runtime) else {
            return;
        };
        crate::placement::publish_provider_feedback_vm_full_call(
            self.receipt,
            self.feedback,
            prelude,
            lease,
            self.command_contribution,
        );
    }
}

/// Native output publication supplies its linear right only after adopting
/// every output. This carrier owns the real selected receipt, not a timer or
/// a VM installation manufactured from matching metadata.
pub(super) struct NativeFusionPublication {
    pub(super) receipt: ProviderOperationReceipt,
    pub(super) feedback: LocalPlacementFeedback,
    pub(super) command_contribution: Option<CommandProviderContribution>,
    pub(super) identity: (
        runmat_types::RegionId,
        runmat_runtime::native::NativeSiteRequest,
    ),
}

impl runmat_runtime::numeric_region::service::NativeNumericRegionPublication
    for NativeFusionPublication
{
    fn complete(
        self: Box<Self>,
        runtime: &RuntimeContext,
        right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
    ) {
        if !right.is_current(runtime)
            || !right.matches(self.identity.0, self.identity.1)
            || !self.feedback.is_current()
        {
            return;
        }
        let Self {
            receipt,
            feedback,
            command_contribution,
            ..
        } = *self;
        let Some(prelude) = receipt.default_clock_prelude_since(right.started()) else {
            return;
        };
        if right.is_current(&feedback.runtime) {
            crate::placement::publish_provider_feedback_full_call(
                receipt,
                feedback,
                prelude,
                command_contribution,
            );
        }
    }
}

/// A placement choice is not an execution failure. The caller runs the shared
/// bytecode path and retains this observation until that region actually exits.
pub enum FusionExecutionOutcome<T> {
    /// Provider dispatch produced a value/handle. This does not establish
    /// completion of asynchronous device work or provide execution timing.
    Executed(T),
    SharedRuntime(FusionCpuObservation),
}

impl<T> FusionExecutionOutcome<T> {
    /// Decline an optimization before effects when no implementation proves
    /// the authored semantics. This grants no selected CPU execution, input
    /// readback, completion or placement-feedback authority.
    pub fn implementation_unavailable() -> Self {
        Self::SharedRuntime(FusionCpuObservation::new(None))
    }
}

/// Measures completion of a shared-runtime bytecode region on the host.
/// Inner calls may submit provider work; this elapsed time is not evidence of
/// completion of that device work and must not be assigned to a GPU candidate.
pub struct FusionCpuObservation {
    feedback: Option<Box<LocalPlacementFeedback>>,
    started: Instant,
    vm_completion: Option<Box<VmCpuCompletion>>,
    native_kind: Option<runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind>,
    native_original_provider: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    native_host_admission: Option<NativeHostAdmission>,
}

struct NativeHostAdmission {
    backing: runmat_execution::resource::ExactShared<NativeHostAdmissionBacking>,
}
struct NativeHostAdmissionBacking {
    admission: super::elementwise::host_readback::Admission,
    // All captured shape/spec/source vector backing retires before its quote.
    _receipt: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
    _claim: Option<std::sync::Arc<dyn runmat_execution::resource::RetainedHostAllocationClaim>>,
}
pub(super) fn native_host_admission_retained_bytes() -> Option<u64> {
    let bytes = u64::try_from(runmat_execution::resource::ExactShared::<
        NativeHostAdmissionBacking,
    >::requested_bytes())
    .ok()?;
    runmat_runtime::context::host_allocation::HostAllocationPlan::retained_metadata_bytes(bytes)
        .ok()
}
impl NativeHostAdmission {
    fn new(
        runtime: &RuntimeContext,
        admission: super::elementwise::host_readback::Admission,
        receipt: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Self> {
        use runmat_accelerate_api::ProviderExecutionError;
        use runmat_runtime::context::host_allocation::{
            HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
        };
        let mut backing = NativeHostAdmissionBacking {
            admission,
            _receipt: receipt,
            _claim: None,
        };
        let bytes = u64::try_from(runmat_execution::resource::ExactShared::<
            NativeHostAdmissionBacking,
        >::requested_bytes())
        .map_err(|_| ProviderExecutionError::from_source(HostAllocationError::Overflow))?;
        let retained = native_host_admission_retained_bytes()
            .ok_or_else(|| ProviderExecutionError::from_source(HostAllocationError::Overflow))?;
        let receipt = HostAllocationReceipt::admit(
            Some(runtime),
            HostAllocationPlan::new(0, retained).map_err(ProviderExecutionError::from_source)?,
        )
        .map_err(ProviderExecutionError::from_source)?;
        backing._claim = receipt
            .as_ref()
            .map(|receipt| receipt.claim_retained_metadata(runtime, bytes))
            .transpose()
            .map_err(ProviderExecutionError::from_source)?;
        runmat_execution::resource::ExactShared::try_new(backing)
            .map(|backing| Self { backing })
            .map_err(|_| {
                ProviderExecutionError::from_source(HostAllocationError::Numeric(
                    runmat_value::ExactHostNumericAllocationError::AllocationFailed,
                ))
            })
    }
}

/// The installed VM right is deliberately linear and relatively large. Only
/// an actual VM-bound host selection allocates this state; ordinary shared
/// bytecode outcomes keep the small observation envelope inline.
struct VmCpuCompletion {
    right: VmFusionCompletionRight,
    host_route: Option<VmFusionHostRoute>,
    runtime: Option<RuntimeContext>,
    readback: Option<CpuReadbackPlan>,
    outputs: Option<super::elementwise::host_readback::HostOutputPlan>,
    prepared: Option<super::host_readback::PreparedVmHostInputs>,
}

impl FusionCpuObservation {
    pub(super) fn new(feedback: Option<Box<LocalPlacementFeedback>>) -> Self {
        Self {
            feedback,
            started: Instant::now(),
            vm_completion: None,
            native_kind: None,
            native_original_provider: None,
            native_host_admission: None,
        }
    }

    pub(super) fn native_selected(
        feedback: Option<Box<LocalPlacementFeedback>>,
        native_kind: Option<runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind>,
        native_original_provider: Option<&'static dyn runmat_accelerate_api::AccelProvider>,
    ) -> Self {
        Self {
            feedback,
            native_kind,
            native_original_provider,
            native_host_admission: None,
            started: Instant::now(),
            vm_completion: None,
        }
    }

    pub(super) fn attach_native_host_admission(
        &mut self,
        runtime: &RuntimeContext,
        admission: Option<super::elementwise::host_readback::Admission>,
        receipt: Option<runmat_runtime::context::host_allocation::HostAllocationReceipt>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<()> {
        if self.native_kind
            != Some(runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Generic)
        {
            // Preparation metadata physically retires before its original
            // funding receipt on vectorized/original/declined selections too.
            drop(admission);
            drop(receipt);
            return Ok(());
        }
        self.native_host_admission = admission
            .map(|admission| NativeHostAdmission::new(runtime, admission, receipt))
            .transpose()?;
        Ok(())
    }

    pub(super) async fn into_native_selection(
        mut self,
        runtime: &RuntimeContext,
        right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        program: &runmat_runtime::numeric_region::service::NativeNumericRegionProgramOwner,
        inputs: &[Value],
    ) -> runmat_accelerate_api::ProviderExecutionResult<
        runmat_runtime::numeric_region::service::NativeNumericRegionOutcome,
    > {
        use runmat_runtime::numeric_region::service::{
            NativeNumericRegionCpuObservation, NativeNumericRegionOutcome,
        };
        if let Some(provider) = self.native_original_provider.take() {
            let observation = native_original::NativeOriginalObservation::admit(
                runtime,
                provider,
                &right,
                self.feedback.take(),
            )?;
            return Ok(NativeNumericRegionOutcome::Original { right, observation });
        }
        let Some(kind) = self.native_kind.take() else {
            self.feedback = None;
            return Ok(NativeNumericRegionOutcome::Ineligible);
        };
        let host = if kind
            == runmat_runtime::numeric_region::service::NativeNumericRegionCpuKind::Generic
        {
            let captured = self.native_host_admission.take().ok_or_else(|| {
                runmat_accelerate_api::ProviderExecutionError::from_source(
                    runmat_runtime::runtime_error::semantic_error(
                        "RunMat:Placement:InvalidNativeRegion",
                        "selected native source-host candidate has no resource/input owner",
                    ),
                )
            })?;
            Some(prepare_native_host(runtime, program, &right, inputs, captured).await?)
        } else {
            None
        };
        Ok(NativeNumericRegionOutcome::NativeCpu {
            kind,
            right,
            host,
            observation: self
                .feedback
                .take()
                .map(|feedback| feedback as Box<dyn NativeNumericRegionCpuObservation>),
        })
    }

    /// Only the selected VM reduction can attach its one-shot installation
    /// right. Missing route proof suppresses feedback but not bytecode.
    pub fn with_vm_host_route(
        self,
        right: VmFusionCompletionRight,
        runtime: Option<&RuntimeContext>,
        bindings: Option<&[runmat_builtins::BuiltinBindingIdentity]>,
    ) -> Self {
        self.with_vm_host_route_readback(right, runtime, bindings, None)
    }

    pub fn with_vm_host_operations(
        self,
        right: VmFusionCompletionRight,
        runtime: Option<&RuntimeContext>,
        operations: Option<Vec<runmat_runtime::context::VmFusionHostOperation>>,
    ) -> Self {
        self.with_vm_host_operations_readback(right, runtime, operations, None, None)
    }

    pub(crate) fn with_vm_host_operations_readback(
        mut self,
        right: VmFusionCompletionRight,
        runtime: Option<&RuntimeContext>,
        operations: Option<Vec<runmat_runtime::context::VmFusionHostOperation>>,
        readback: Option<CpuReadbackPlan>,
        outputs: Option<super::elementwise::host_readback::HostOutputPlan>,
    ) -> Self {
        // The VM right authenticates execution. A missing price/measurement
        // permit suppresses feedback, not an already admitted host readback.
        let host_route = runtime
            .filter(|_| {
                self.feedback
                    .as_ref()
                    .is_none_or(|feedback| feedback.is_current())
            })
            .and_then(|runtime| {
                operations.and_then(|operations| right.host_route_operations(runtime, operations))
            });
        self.vm_completion = Some(Box::new(VmCpuCompletion {
            right,
            host_route,
            runtime: runtime.cloned(),
            readback,
            outputs,
            prepared: None,
        }));
        self
    }

    pub(crate) fn with_vm_host_route_readback(
        mut self,
        right: VmFusionCompletionRight,
        runtime: Option<&RuntimeContext>,
        bindings: Option<&[runmat_builtins::BuiltinBindingIdentity]>,
        readback: Option<CpuReadbackPlan>,
    ) -> Self {
        let host_route = runtime
            .filter(|_| {
                self.feedback
                    .as_ref()
                    .is_none_or(|feedback| feedback.is_current())
            })
            .and_then(|runtime| {
                bindings.and_then(|bindings| right.host_route(runtime, bindings.to_vec()))
            });
        self.vm_completion = Some(Box::new(VmCpuCompletion {
            right,
            host_route,
            runtime: readback.as_ref().and_then(|_| runtime.cloned()),
            readback,
            outputs: None,
            prepared: None,
        }));
        self
    }

    pub fn vm_host_route(&self) -> Option<VmFusionHostRoute> {
        self.vm_completion
            .as_ref()
            .and_then(|vm| vm.host_route.clone())
    }

    /// Revalidate the exact selected right and any feedback permit before the
    /// first readback. All stage results remain private until the VM atomically
    /// replaces its checked operand slice.
    /// Called after an ordinary load has completed all normal refresh and
    /// language checks. A refreshed source mismatch is a terminal post-effect
    /// error, never permission to replay the already prepared region.
    pub fn substitute_vm_host_load(
        &self,
        pc: usize,
        binding: &crate::graph::VarBinding,
        actual: &Value,
    ) -> Result<Option<Value>, Box<runmat_runtime::RuntimeError>> {
        let Some(vm) = self.vm_completion.as_ref() else {
            return Ok(None);
        };
        let Some(prepared) = vm.prepared.as_ref() else {
            return Ok(None);
        };
        if !vm_readback_current(vm, self.feedback.as_deref()) {
            return Err(Box::new(
                runmat_runtime::build_runtime_error("selected host route expired at ordinary load")
                    .with_identifier("RunMat:Placement:StaleHostReadback")
                    .build(),
            ));
        }
        prepared.substitute_load(pc, binding, actual)
    }

    pub async fn prepare_vm_host_inputs(
        &mut self,
        stack: &[Value],
    ) -> Result<Option<Vec<(usize, Value)>>, CpuReadbackFailure> {
        let Some(vm) = self.vm_completion.as_mut() else {
            return Ok(None);
        };
        if vm.readback.is_none() && vm.outputs.is_none() {
            return Ok(None);
        }
        let feedback = self.feedback.as_deref();
        if !vm_readback_current(vm, feedback) {
            return Err(CpuReadbackFailure::BeforeEffect(
                runmat_runtime::build_runtime_error(
                    "selected VM host route expired before input preparation",
                )
                .build(),
            ));
        }
        prepare_vm_outputs(vm).map_err(|error| *error)?;
        if vm.readback.is_none() {
            return Ok(None);
        }
        let plan = vm.readback.take().expect("checked readback plan");
        let mut inputs = prepare_selected_host_inputs(plan, stack, vm.runtime.as_ref()).await?;
        if !vm_readback_current(vm, feedback) {
            return Err(CpuReadbackFailure::AfterEffect(
                runmat_runtime::build_runtime_error(
                    "selected VM host route expired during input preparation",
                )
                .build(),
            ));
        }
        let entry = std::mem::take(&mut inputs.entry);
        vm.prepared = Some(inputs);
        Ok(Some(entry))
    }

    pub async fn prepare_vm_host_slice(
        &mut self,
        stack: &[Value],
    ) -> Result<Option<Vec<(usize, Value)>>, CpuReadbackFailure> {
        let Some(vm) = self.vm_completion.as_mut() else {
            return Ok(None);
        };
        if vm.readback.is_none() && vm.outputs.is_none() {
            return Ok(None);
        }
        if !vm_readback_current(vm, self.feedback.as_deref()) {
            return Err(CpuReadbackFailure::BeforeEffect(
                runmat_runtime::build_runtime_error(
                    "selected VM host route or placement permit expired before readback",
                )
                .with_identifier("RunMat:Placement:StaleHostReadback")
                .build(),
            ));
        }
        prepare_vm_outputs(vm).map_err(|error| *error)?;
        if vm.readback.is_none() {
            return Ok(None);
        }
        let plan = vm.readback.take().expect("checked readback plan");
        let mut inputs = prepare_selected_host_inputs(plan, stack, vm.runtime.as_ref()).await?;
        if !vm_readback_current(vm, self.feedback.as_deref()) {
            return Err(CpuReadbackFailure::AfterEffect(
                runmat_runtime::build_runtime_error(
                    "selected VM host route or placement permit expired during readback",
                )
                .with_identifier("RunMat:Placement:StaleHostReadback")
                .build(),
            ));
        }
        let entry = std::mem::take(&mut inputs.entry);
        vm.prepared = Some(inputs);
        Ok(Some(entry))
    }

    /// VM-owned FullCall host completion begins before placement planning and
    /// follows only exact normal bytecode publication.
    pub fn complete_vm(mut self, outer_started: Instant) {
        let Some(vm) = self.vm_completion.take() else {
            self.complete();
            return;
        };
        let VmCpuCompletion {
            right,
            host_route,
            runtime: _,
            readback,
            outputs: _,
            prepared,
        } = *vm;
        // The following ordinary store may have overwritten the last
        // workspace reference to a resident source. Retire our retained
        // references through the existing checked release owner before
        // issuing any successful FullCall receipt. Never retry on failure.
        let mut cleanup_succeeded = true;
        if let Some(prepared) = prepared {
            if let Err(error) = prepared.cleanup() {
                log::warn!("VM host preparation cleanup failed: {error}");
                cleanup_succeeded = false;
            }
        }
        if let Some(readback) = readback {
            if let Err(error) = readback.cleanup() {
                log::warn!("VM host readback cleanup failed: {error}");
                cleanup_succeeded = false;
            }
        }
        let (Some(route), Some(feedback)) = (host_route, self.feedback.take()) else {
            return;
        };
        if !cleanup_succeeded
            || !route.complete()
            || !feedback.is_current()
            || feedback.runtime.is_cancelled()
        {
            return;
        }
        let Some(_lease) = right.completion_lease(&feedback.runtime) else {
            return;
        };
        feedback.observe(duration_ns_saturating(outer_started.elapsed()), true);
    }

    /// Only the owner of the complete bytecode region may report success.
    /// Success means host bytecode exited, not that nested GPU queues drained.
    pub fn complete(mut self) {
        if self.vm_completion.is_some() {
            // The VM-bound route requires its exact instruction and
            // publication proof; the legacy completion API cannot issue it.
            self.feedback = None;
            return;
        }
        let succeeded = self
            .feedback
            .as_ref()
            .is_none_or(|feedback| !feedback.runtime.is_cancelled());
        self.observe(succeeded);
    }

    fn observe(&mut self, succeeded: bool) {
        let Some(feedback) = self.feedback.take() else {
            return;
        };
        feedback.observe(duration_ns_saturating(self.started.elapsed()), succeeded);
    }
}

impl runmat_runtime::numeric_region::service::NativeNumericRegionCpuObservation
    for LocalPlacementFeedback
{
    fn finish(
        self: Box<Self>,
        runtime: &RuntimeContext,
        right: runmat_runtime::numeric_region::service::NativeNumericRegionRight,
        succeeded: bool,
    ) {
        if right.is_current(runtime)
            && self.runtime.local_identity() == runtime.local_identity()
            && self.signature.region == Some(right.identity().0)
            && self.is_current()
        {
            (*self).observe(
                duration_ns_saturating(right.started().elapsed()).max(1),
                succeeded,
            );
        }
    }
}

impl Drop for FusionCpuObservation {
    fn drop(&mut self) {
        if self.vm_completion.is_some() {
            self.feedback = None;
        } else {
            self.observe(false);
        }
    }
}

fn vm_readback_current(vm: &VmCpuCompletion, feedback: Option<&LocalPlacementFeedback>) -> bool {
    vm.runtime.as_ref().is_some_and(|runtime| {
        let revision = vm.right.planning_revision(runtime);
        !runtime.is_cancelled()
            && vm
                .host_route
                .as_ref()
                .is_some_and(|route| route.current_for(runtime))
            && revision.is_some()
            && feedback.is_none_or(|feedback| {
                feedback.is_current() && revision.as_ref() == Some(&feedback.signature.revision)
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion_exec::FusionPlacementSelection;
    use crate::placement::{FusionPlacementObserver, PlacementSession};
    use runmat_execution::{Digest, PlacementRevision, PlacementSignature};
    use runmat_runtime::context::{RuntimeContext, RuntimeServicePorts};
    use std::rc::Rc;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn cpu_observation_envelope_does_not_inline_the_vm_right() {
        assert!(
            std::mem::size_of::<FusionCpuObservation>()
                < std::mem::size_of::<VmFusionCompletionRight>()
        );
        assert!(
            std::mem::size_of::<FusionExecutionOutcome<()>>()
                < std::mem::size_of::<VmFusionCompletionRight>()
        );
        assert!(FusionCpuObservation::new(None).vm_completion.is_none());
    }

    fn ticket(
        candidate: &str,
        kind: runmat_execution::ExecutionCandidateKind,
    ) -> (LocalPlacementFeedback, Rc<PlacementSession>) {
        let session = Rc::new(PlacementSession::default());
        let runtime = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_placement(session.clone()));
        let digest = || Digest::sha256(b"fusion-observation-test");
        let signature = PlacementSignature {
            region: None,
            operation: "fusion.observation.test".into(),
            runtime_facts: digest(),
            provider_provenance: None,
            graph_output_authority: None,
            revision: PlacementRevision {
                program: None,
                catalog: digest(),
                compiler: digest(),
                authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest()),
                policy: digest(),
            },
        };
        (
            LocalPlacementFeedback {
                runtime,
                authority: crate::placement::LocalPlacementFeedbackAuthority::Graph(
                    session.test_graph_observation_permit(signature.clone(), candidate, kind),
                ),
                signature,
                candidate: candidate.into(),
                kind,
            },
            session,
        )
    }

    #[test]
    fn cpu_observation_records_success_once_only_after_completion() {
        let (feedback, session) = ticket(
            "cpu.shared.test",
            runmat_execution::ExecutionCandidateKind::SharedRuntime,
        );
        let observation = FusionCpuObservation::new(Some(Box::new(feedback)));
        assert!(session.profile_snapshot().feedback.is_empty());
        observation.complete();
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback.len(), 1);
        assert_eq!(snapshot.feedback[0].samples, 1);
        assert_eq!(snapshot.feedback[0].failures, 0);
    }

    #[test]
    fn revoked_cpu_selection_exits_without_learning_a_failure_or_success() {
        let (feedback, session) = ticket(
            "cpu.shared.revoked",
            runmat_execution::ExecutionCandidateKind::SharedRuntime,
        );
        let authority = match feedback.signature.revision.authority {
            runmat_execution::PlacementExecutionAuthorityRevision::Provider(digest) => digest,
            _ => unreachable!("fixture provider authority"),
        };
        let observation = FusionCpuObservation::new(Some(Box::new(feedback)));
        session.invalidate(runmat_execution::PlacementInvalidation::Provider { digest: authority });
        observation.complete();
        assert!(session.profile_snapshot().feedback.is_empty());
    }

    #[test]
    fn abandoned_cpu_observation_never_records_success() {
        let (feedback, session) = ticket(
            "cpu.shared.test",
            runmat_execution::ExecutionCandidateKind::SharedRuntime,
        );
        drop(FusionCpuObservation::new(Some(Box::new(feedback))));
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback[0].samples, 0);
        assert_eq!(snapshot.feedback[0].failures, 1);
    }

    #[test]
    fn cancelled_future_drops_cpu_ticket_without_a_success_sample() {
        let (feedback, session) = ticket(
            "cpu.shared.test",
            runmat_execution::ExecutionCandidateKind::SharedRuntime,
        );
        let observation = FusionCpuObservation::new(Some(Box::new(feedback)));
        let future = async move {
            std::future::pending::<()>().await;
            observation.complete();
        };
        drop(future);
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback[0].samples, 0);
        assert_eq!(snapshot.feedback[0].failures, 1);
    }

    #[test]
    fn gpu_submission_is_not_completed_execution_feedback() {
        let (feedback, session) = ticket(
            "gpu.fusion.test",
            runmat_execution::ExecutionCandidateKind::ProviderFusion,
        );
        FusionPlacementSelection::new(
            FusionPlacementObserver::new(None),
            Some(feedback),
            &crate::simple_provider::InProcessProvider::new(),
            None,
        )
        .submitted();
        assert!(session.profile_snapshot().feedback.is_empty());
    }

    #[test]
    fn failed_gpu_submission_retains_failure_feedback() {
        let (feedback, session) = ticket(
            "gpu.fusion.test",
            runmat_execution::ExecutionCandidateKind::ProviderFusion,
        );
        drop(FusionPlacementSelection::new(
            FusionPlacementObserver::new(None),
            Some(feedback),
            &crate::simple_provider::InProcessProvider::new(),
            None,
        ));
        let snapshot = session.profile_snapshot();
        assert_eq!(snapshot.feedback[0].samples, 0);
        assert_eq!(snapshot.feedback[0].failures, 1);
    }
}
