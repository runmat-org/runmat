use super::RuntimeContext;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt,
};
use crate::RuntimeError;
use runmat_thread_local::runmat_thread_local;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

#[path = "scope/deferred_future.rs"]
mod deferred_future;
#[path = "scope/future_owner.rs"]
mod future_owner;
pub use deferred_future::DeferredRuntimeFuture;
mod unselected;
pub(crate) use unselected::PreparedUnselectedScope;
mod inline;
mod lazy;
pub(crate) use inline::{with_original_scope, RuntimeScopeFactory};
pub(crate) use lazy::LazyRuntimeScope;

runmat_thread_local! {
    static ACTIVE_CONTEXTS: RefCell<Vec<Option<RuntimeContext>>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn active_runtime_context() -> Option<RuntimeContext> {
    ACTIVE_CONTEXTS.with(|contexts| contexts.borrow().last().cloned().flatten())
}

/// Read poll-local owners without cloning the complete invocation context.
/// The reader must not enter another runtime scope while the stack is borrowed.
pub(super) fn with_active_runtime_context<T>(read: impl FnOnce(Option<&RuntimeContext>) -> T) -> T {
    ACTIVE_CONTEXTS.with(|contexts| read(contexts.borrow().last().and_then(Option::as_ref)))
}

pub(super) fn execution_provider_authorization(
    admission: runmat_accelerate_api::ExecutionProviderAdmission<'_>,
) -> runmat_accelerate_api::ExecutionProviderAuthorization {
    use runmat_accelerate_api::ExecutionProviderAuthorization as Authorization;
    use runmat_accelerate_api::{ExecutionDeviceEffect, ExecutionProviderAdmission};

    let selected = runmat_execution::profiling::current_command_profiling();
    let context = active_runtime_context();
    if let Some(selected) = &selected {
        let matching = context
            .as_ref()
            .and_then(RuntimeContext::command_profiling)
            .is_some_and(|owner| selected.same_command(owner));
        if !matching {
            selected.mark_unproven_device_route();
        }
    }
    if let Some(context) = &context {
        if let Some(context_owner) = context.command_profiling() {
            if !selected
                .as_ref()
                .is_some_and(|owner| owner.same_command(context_owner))
            {
                context_owner.mark_unproven_device_route();
            }
        }
    }

    let control = matches!(&admission, ExecutionProviderAdmission::DeviceControl(_));
    if let Some(policy) = context
        .as_ref()
        .and_then(RuntimeContext::acceleration_policy)
    {
        let permitted = match &admission {
            ExecutionProviderAdmission::Provider(provider)
            | ExecutionProviderAdmission::DeviceControl(provider) => {
                policy.permits_provider(*provider)
            }
            ExecutionProviderAdmission::NumericReadback { provider, handle } => {
                provider.device_id() == handle.device_id()
                    && handle.allocation_lease().validate_live().is_ok()
            }
            ExecutionProviderAdmission::DeviceAccess(_)
            | ExecutionProviderAdmission::CanonicalWgpuSubmit(_) => {
                policy.enabled && policy.provider != super::RuntimeProviderPreference::InProcess
            }
            ExecutionProviderAdmission::DeviceEffect(_) => true,
        };
        if !permitted {
            return Authorization::Denied;
        }
    }
    match admission {
        ExecutionProviderAdmission::DeviceAccess(access) => {
            let Some(context) = context else {
                return Authorization::Unrestricted;
            };
            use super::gpu_device::GpuDeviceSelection;
            let selected = match context.gpu_device_selection() {
                GpuDeviceSelection::Inherit => {
                    context.execution_assignment().is_none()
                        || context.command_device_provider().is_some_and(|provider| {
                            runmat_accelerate_api::device_control_provider_is_authorized(provider)
                                && provider
                                    .wgpu_device_access()
                                    .is_some_and(|expected| expected.same_device(access))
                        })
                }
                GpuDeviceSelection::Selected(provider) => {
                    runmat_accelerate_api::device_control_provider_is_authorized(provider)
                        && provider
                            .wgpu_device_access()
                            .is_some_and(|expected| expected.same_device(access))
                }
                GpuDeviceSelection::Deselected
                | GpuDeviceSelection::Resetting(_)
                | GpuDeviceSelection::ResetFailed(_) => false,
            };
            if selected {
                Authorization::Allowed
            } else {
                Authorization::Denied
            }
        }
        ExecutionProviderAdmission::CanonicalWgpuSubmit(access) => {
            let matching = context
                .as_ref()
                .and_then(RuntimeContext::command_device_access)
                .is_some_and(|expected| expected.same_device(access) && !access.has_raw_export());
            if !matching {
                if let Some(owner) = selected.or_else(|| {
                    context
                        .as_ref()
                        .and_then(RuntimeContext::command_profiling)
                        .cloned()
                }) {
                    owner.mark_unproven_device_route();
                }
            }
            Authorization::Unrestricted
        }
        ExecutionProviderAdmission::DeviceEffect(effect) => {
            if let Some(owner) = selected
                .or_else(|| context.and_then(|context| context.command_profiling().cloned()))
            {
                match effect {
                    ExecutionDeviceEffect::ImplicitQueueTransfer => {
                        owner.mark_unproven_device_time()
                    }
                    ExecutionDeviceEffect::RawWgpuExport
                    | ExecutionDeviceEffect::RawNativeExport
                    | ExecutionDeviceEffect::PlotDeviceEmission
                    | ExecutionDeviceEffect::ForeignExecution => owner.mark_unproven_device_route(),
                }
            }
            Authorization::Unrestricted
        }
        ExecutionProviderAdmission::Provider(provider)
        | ExecutionProviderAdmission::DeviceControl(provider)
        | ExecutionProviderAdmission::NumericReadback { provider, .. } => {
            let mut authorization = context
                .as_ref()
                .and_then(RuntimeContext::execution_assignment)
                .map_or(Authorization::Unrestricted, |assignment| {
                    if assignment.resources.accelerator_leases.iter().any(|lease| {
                        runmat_accelerate_api::provider_matches_execution_lease(provider, lease)
                    }) {
                        Authorization::Allowed
                    } else {
                        Authorization::Denied
                    }
                });
            if !control && authorization != Authorization::Denied {
                if let Some(context) = &context {
                    use super::gpu_device::GpuDeviceSelection;
                    authorization = match context.gpu_device_selection() {
                        GpuDeviceSelection::Inherit => authorization,
                        GpuDeviceSelection::Selected(selected)
                            if runmat_accelerate_api::same_provider_instance(
                                selected, provider,
                            ) =>
                        {
                            Authorization::Allowed
                        }
                        _ => Authorization::Denied,
                    };
                }
            }
            if authorization != Authorization::Denied {
                if let Some(owner) = selected {
                    let canonical = context
                        .as_ref()
                        .and_then(RuntimeContext::command_device_provider)
                        .is_some_and(|expected| {
                            runmat_accelerate_api::same_provider_instance(provider, expected)
                        });
                    if !canonical {
                        owner.mark_unproven_device_route();
                    }
                }
            }
            authorization
        }
    }
}

#[must_use]
pub struct RuntimeContextGuard {
    state_identity: Option<*const super::RuntimeContextState>,
    _display_format: runmat_value::DisplayFormatSelectionGuard,
    _command_profiling: runmat_execution::profiling::CommandProfilingSelectionGuard,
    _tape: crate::builtins::deep_learning::autodiff::TapePollGuard,
}

impl RuntimeContextGuard {
    fn enter_unselected() -> Self {
        let display = runmat_value::select_optional_display_format_owner(None);
        let profiling = runmat_execution::profiling::select_command_profiling(None);
        let tape = crate::builtins::deep_learning::autodiff::TapePollGuard::select(None);
        ACTIVE_CONTEXTS.with(|contexts| contexts.borrow_mut().push(None));
        Self {
            state_identity: None,
            _display_format: display,
            _command_profiling: profiling,
            _tape: tape,
        }
    }
    pub(crate) fn enter(context: RuntimeContext) -> Self {
        let state_identity = Some(context.state_identity());
        let display_format =
            runmat_value::select_display_format_owner(context.display_format_owner());
        let command_profiling = runmat_execution::profiling::select_command_profiling(
            context.command_profiling().cloned(),
        );
        let tape = crate::builtins::deep_learning::autodiff::TapePollGuard::select(
            context.tape_poll_authority(),
        );
        ACTIVE_CONTEXTS.with(|contexts| contexts.borrow_mut().push(Some(context)));
        Self {
            state_identity,
            _display_format: display_format,
            _command_profiling: command_profiling,
            _tape: tape,
        }
    }
}

impl Drop for RuntimeContextGuard {
    fn drop(&mut self) {
        ACTIVE_CONTEXTS.with(|contexts| {
            let popped = contexts.borrow_mut().pop();
            debug_assert!(
                popped.is_some(),
                "runtime context guard must own a stack entry"
            );
            debug_assert_eq!(
                popped
                    .as_ref()
                    .and_then(Option::as_ref)
                    .map(RuntimeContext::state_identity),
                self.state_identity,
                "runtime contexts must unwind in stack order"
            );
        });
    }
}

/// Future adapter that activates the same context for every poll and while the
/// inner future is destroyed. It never keeps ambient state installed across a
/// yield, so independently polled sessions cannot observe one another.
pub struct ContextFuture<F> {
    scope: Option<PollFutureScope>,
    future: Option<Pin<Box<F>>>,
}

/// One captured poll authority and its admitted scratch lifetime, shared by
/// ordinary and deferred future owners without duplicating the context.
struct PollFutureScope {
    context: RuntimeContext,
    _future_scratch: Option<std::sync::Arc<super::host_effect::ManagedHostEffectStageReservation>>,
}

impl<F> ContextFuture<F> {
    pub(super) fn new(mut context: RuntimeContext, future: Pin<Box<F>>) -> Self {
        context.capture_poll_projections();
        Self {
            scope: Some(PollFutureScope {
                context,
                _future_scratch: None,
            }),
            future: Some(future),
        }
    }
}

/// A complete poll-local context snapshot prepared before a queued future is
/// constructed. Managed snapshots retain their exact projection owners and
/// admit the concrete future owner before its factory is invoked.
pub struct PreparedRuntimeScope {
    context: RuntimeContext,
}

impl PreparedRuntimeScope {
    /// Construct an unboxed result child under this already captured poll
    /// authority. Its caller admits the complete child at the existing
    /// continuation owner before invoking this factory; no second Box is made.
    /// The explicit child context and every poll/drop share this snapshot.
    pub fn construct_result_unboxed<F, O>(
        self,
        make_future: impl FnOnce(RuntimeContext) -> F,
    ) -> impl Future<Output = Result<O, RuntimeError>>
    where
        F: Future<Output = Result<O, RuntimeError>>,
    {
        let future = {
            let _guard = self.enter();
            make_future(self.context.clone())
        };
        inline::ScopedFuture::from_prepared(self, future)
    }

    pub(crate) fn enter(&self) -> RuntimeContextGuard {
        RuntimeContextGuard::enter(self.context.clone())
    }
    /// Admit the concrete future owner before its factory is invoked.
    pub fn prepare_future<F: Future>(self) -> Result<PreparedRuntimeFutureScope<F>, RuntimeError> {
        let bytes = u64::try_from(std::mem::size_of::<F>()).map_err(|_| {
            scope_allocation_error(
                &self.context,
                "poll future allocation geometry overflowed",
                HostAllocationError::Overflow,
            )
        })?;
        let receipt = HostAllocationReceipt::admit(
            Some(&self.context),
            HostAllocationPlan::new(bytes, 0).map_err(|source| {
                scope_allocation_error(
                    &self.context,
                    "poll future allocation geometry overflowed",
                    source,
                )
            })?,
        )
        .map_err(|source| {
            scope_allocation_error(&self.context, "poll future admission failed", source)
        })?;
        let scratch = receipt
            .as_ref()
            .map(HostAllocationReceipt::retain_scratch_for_cpu_work);
        if let Some(receipt) = &receipt {
            receipt.checkpoint(&self.context).map_err(|source| {
                scope_allocation_error(&self.context, "poll future authority changed", source)
            })?;
        }
        let cancellation = self.context.cancellation();
        Ok(PreparedRuntimeFutureScope {
            scope: PollFutureScope {
                context: self.context,
                _future_scratch: scratch,
            },
            preparation: FuturePreparation {
                cancellation,
                receipt,
            },
            _future: std::marker::PhantomData,
        })
    }
}

/// Exact preparation for one concrete scoped future allocation.
pub struct PreparedRuntimeFutureScope<F> {
    scope: PollFutureScope,
    preparation: FuturePreparation,
    _future: std::marker::PhantomData<fn() -> F>,
}

/// Admission state is consumed once before the factory runs; the common poll
/// scope remains with the running future and its exact scratch owner.
struct FuturePreparation {
    cancellation: runmat_execution::cancellation::CancellationToken,
    receipt: Option<HostAllocationReceipt>,
}

impl<F: Future> PreparedRuntimeFutureScope<F> {
    /// Defer construction until the queue has unwound its caller's poll.
    pub fn defer_result<M, T>(self, make_future: M) -> DeferredRuntimeFuture<F, M, T>
    where
        F: Future<Output = Result<T, RuntimeError>>,
        M: FnOnce() -> F,
    {
        DeferredRuntimeFuture::new(self, make_future)
    }

    pub fn try_scope_with(
        self,
        make_future: impl FnOnce() -> F,
    ) -> Result<ContextFuture<F>, RuntimeError> {
        let Self {
            scope,
            preparation,
            _future: _,
        } = self;
        let future = preparation.try_construct(&scope, make_future)?;
        Ok(ContextFuture {
            scope: Some(scope),
            future: Some(future),
        })
    }
}

impl FuturePreparation {
    fn try_construct<F: Future>(
        self,
        scope: &PollFutureScope,
        make_future: impl FnOnce() -> F,
    ) -> Result<Pin<Box<F>>, RuntimeError> {
        let Self {
            cancellation,
            receipt,
        } = self;
        let mut make_future = Some(make_future);
        if let Err(source) =
            checkpoint_unbounded_factory(&scope.context, &cancellation, receipt.is_none())
        {
            inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                future_owner::drop_value_in_context(
                    &scope.context,
                    make_future.take().expect("poll future factory"),
                )
            });
            return Err(source);
        }
        let future = if let Some(receipt) = &receipt {
            if let Err(source) = receipt.checkpoint(&scope.context) {
                inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                    future_owner::drop_value_in_context(
                        &scope.context,
                        make_future.take().expect("poll future factory"),
                    )
                });
                return Err(scope_allocation_error(
                    &scope.context,
                    "poll future authority changed",
                    source,
                ));
            }
            if let Err(error) = inline::check_captured(&scope.context, Some(&cancellation)) {
                inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                    future_owner::drop_value_in_context(
                        &scope.context,
                        make_future.take().expect("poll future factory"),
                    )
                });
                return Err(error);
            }
            let allocated =
                inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                    future_owner::try_pin_in_context(
                        &scope.context,
                        make_future.take().expect("poll future factory"),
                    )
                });
            if let Err(source) = receipt.checkpoint(&scope.context) {
                if let Ok(future) = allocated {
                    inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                        future_owner::drop_in_context(&scope.context, future)
                    });
                }
                return Err(scope_allocation_error(
                    &scope.context,
                    "poll future authority changed",
                    source,
                ));
            }
            allocated.map_err(|source| {
                scope_allocation_error(
                    &scope.context,
                    "poll future allocation failed",
                    HostAllocationError::Numeric(source),
                )
            })?
        } else {
            let future =
                inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                    future_owner::pin_in_context(
                        &scope.context,
                        make_future.take().expect("poll future factory"),
                    )
                });
            if let Err(source) = checkpoint_unbounded_factory(&scope.context, &cancellation, true) {
                inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                    future_owner::drop_in_context(&scope.context, future)
                });
                return Err(source);
            }
            future
        };
        if let Err(error) = inline::check_captured(&scope.context, Some(&cancellation)) {
            inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
                future_owner::drop_in_context(&scope.context, future)
            });
            return Err(error);
        }
        Ok(future)
    }

    fn drop_unconstructed<M>(self, scope: &mut PollFutureScope, make_future: M) {
        // Retire factory captures and admission state under the same selected
        // authority as construction, including cancellation-before-first-poll.
        let cancellation = self.cancellation.clone();
        let scratch = scope._future_scratch.take();
        inline::with_captured_scope(Some(&scope.context), Some(&cancellation), || {
            future_owner::drop_value_in_context(&scope.context, (make_future, self, scratch))
        });
    }
}

fn checkpoint_unbounded_factory(
    context: &RuntimeContext,
    cancellation: &runmat_execution::cancellation::CancellationToken,
    unbounded: bool,
) -> Result<(), RuntimeError> {
    if !unbounded {
        return Ok(());
    }
    if !cancellation.same_generation(&context.cancellation()) {
        return Err(scope_allocation_error(
            context,
            "poll future authority changed",
            HostAllocationError::StaleAuthority,
        ));
    }
    if cancellation.is_cancelled() {
        return Err(scope_allocation_error(
            context,
            "poll future construction was cancelled",
            HostAllocationError::Cancelled,
        ));
    }
    Ok(())
}

impl RuntimeContext {
    /// Prepare every mutable poll-local projection before a child future can
    /// be published. Unbounded execution preserves the legacy snapshot path;
    /// managed execution owns exact projection claims and scratch headroom.
    pub fn prepare_poll_scope(&self) -> Result<PreparedRuntimeScope, RuntimeError> {
        let policy = HostAllocationReceipt::policy(Some(self)).map_err(|source| {
            scope_allocation_error(self, "poll scope authority is unavailable", source)
        })?;
        if policy == HostAllocationPolicy::Unbounded {
            return Ok(prepare_unbounded_poll_scope(self));
        }

        prepare_managed_poll_scope(self)
    }
}

/// Keep the ordinary projection snapshot independent of the substantially
/// larger managed admission state. This function still runs synchronously at
/// the caller's queue boundary, before the child future is constructed.
fn prepare_unbounded_poll_scope(context: &RuntimeContext) -> PreparedRuntimeScope {
    let mut context = context.clone();
    context.capture_poll_projections();
    PreparedRuntimeScope { context }
}

/// Managed preparation retains its complete exact quote/fence behavior while
/// no longer inflating every ordinary caller's active frame.
fn prepare_managed_poll_scope(
    context: &RuntimeContext,
) -> Result<PreparedRuntimeScope, RuntimeError> {
    let quote = context
        .poll_source_debug_quote()
        .and_then(|source| {
            context
                .poll_local_projection_quote()
                .and_then(|local| source.checked_add(local))
        })
        .ok_or_else(|| {
            scope_allocation_error(
                context,
                "poll scope allocation geometry overflowed",
                HostAllocationError::Overflow,
            )
        })?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        quote.retained_bytes,
        quote.retained_claims,
    )
    .map_err(|source| {
        scope_allocation_error(context, "poll scope allocation geometry overflowed", source)
    })?;
    let plan = HostAllocationPlan::new(quote.scratch_bytes, retained).map_err(|source| {
        scope_allocation_error(context, "poll scope allocation geometry overflowed", source)
    })?;
    let receipt = HostAllocationReceipt::admit(Some(context), plan)
        .map_err(|source| scope_allocation_error(context, "poll scope admission failed", source))?
        .ok_or_else(|| {
            scope_allocation_error(
                context,
                "managed poll scope lost its allocation authority",
                HostAllocationError::MissingManagedOwner,
            )
        })?;
    let permit = receipt.output_permit(context).map_err(|source| {
        scope_allocation_error(context, "poll scope output authority changed", source)
    })?;
    let scratch = receipt.retain_scratch_for_cpu_work();
    let prepared = context
        .clone()
        .with_admitted_poll_source_catalog(&permit, scratch.clone())
        .and_then(|context| context.with_admitted_poll_debug_frames(&permit, scratch.clone()))
        .map_err(HostAllocationError::Numeric)
        .and_then(|context| context.with_admitted_poll_local_projections(&permit, scratch));
    receipt.checkpoint(context).map_err(|source| {
        scope_allocation_error(context, "poll scope authority changed", source)
    })?;
    let context = prepared.map_err(|source| {
        scope_allocation_error(context, "poll scope projection allocation failed", source)
    })?;
    Ok(PreparedRuntimeScope { context })
}

fn scope_allocation_error(
    context: &RuntimeContext,
    message: &'static str,
    source: HostAllocationError,
) -> RuntimeError {
    host_allocation_runtime_error(
        Some(context),
        "RunMat:Runtime:PollScopeAllocation",
        message,
        source,
    )
}

impl PollFutureScope {
    fn poll<F: Future>(
        &self,
        future: &mut Option<Pin<Box<F>>>,
        cx: &mut Context<'_>,
    ) -> Poll<F::Output> {
        let _guard = RuntimeContextGuard::enter(self.context.clone());
        let result = future
            .as_mut()
            .expect("context future polled after completion")
            .as_mut()
            .poll(cx);
        if result.is_ready() {
            // Completion destructors run under the issuing invocation before
            // observing receipts they may complete or invalidate.
            drop(future.take());
            if let Some(placement) = self.context.service_ports().placement() {
                placement.drain_feedback();
            }
        }
        result
    }

    fn drop_future<F>(&self, future: &mut Option<Pin<Box<F>>>) {
        if let Some(future) = future.take() {
            let _guard = RuntimeContextGuard::enter(self.context.clone());
            drop(future);
            if let Some(placement) = self.context.service_ports().placement() {
                placement.drain_feedback();
            }
        }
    }
}

impl<F: Future> Future for ContextFuture<F> {
    type Output = F::Output;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        match &this.scope {
            Some(scope) => scope.poll(&mut this.future, cx),
            None => unselected::poll(&mut this.future, cx),
        }
    }
}

impl<F> Drop for ContextFuture<F> {
    fn drop(&mut self) {
        match &self.scope {
            Some(scope) => scope.drop_future(&mut self.future),
            None => unselected::drop_future(&mut self.future),
        }
    }
}
impl<F> Unpin for ContextFuture<F> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeExecutionStack;
    use crate::execution::RuntimeExecutionService;
    use runmat_execution::profiling::{
        current_command_profiling, CommandProfilingRoot, ProviderObservationTerminal,
    };
    use runmat_types::SourceId;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::Wake;

    struct NoopWake;

    struct ExecutionProvider {
        id: &'static str,
        device_id: u32,
    }

    impl runmat_accelerate_api::AccelProvider for ExecutionProvider {
        fn upload(
            &self,
            _host: &runmat_accelerate_api::HostTensorView,
        ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
            anyhow::bail!("not used")
        }

        fn download<'a>(
            &'a self,
            _handle: &'a runmat_accelerate_api::GpuTensorHandle,
        ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("not used") })
        }

        fn device_info(&self) -> String {
            self.id.into()
        }

        fn device_id(&self) -> u32 {
            self.device_id
        }

        fn device_info_struct(&self) -> runmat_accelerate_api::ApiDeviceInfo {
            runmat_accelerate_api::ApiDeviceInfo {
                device_id: self.device_id,
                name: self.id.into(),
                vendor: "RunMat".into(),
                memory_bytes: Some(4096),
                backend: Some("test".into()),
            }
        }

        fn execution_accelerator_device(
            &self,
            inventory_epoch: u64,
        ) -> anyhow::Result<Option<runmat_execution::resource::AcceleratorDevice>> {
            runmat_accelerate_api::execution_accelerator_device(
                self,
                self.id,
                "1.0.0",
                runmat_accelerate_api::ExecutionProviderContract::GenericComputeV1,
                &format!("test-device-{}", self.device_id),
                &format!("test-device-{}", self.device_id),
                inventory_epoch,
            )
        }
    }

    static LEASED_PROVIDER: ExecutionProvider = ExecutionProvider {
        id: "runmat.test.leased",
        device_id: 901,
    };
    static OTHER_PROVIDER: ExecutionProvider = ExecutionProvider {
        id: "runmat.test.other",
        device_id: 902,
    };

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    #[test]
    fn nested_runtime_guards_restore_exact_command_profiling_owner() {
        let first_receipt = CommandProfilingRoot::new();
        let second_receipt = CommandProfilingRoot::new();
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_command_profiling(first_receipt.receipt());
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_command_profiling(second_receipt.receipt());
        let _first = first.enter();
        let second_guard = second.enter();
        current_command_profiling()
            .expect("second selected")
            .begin_provider_operation()
            .expect("second contribution")
            .complete(ProviderObservationTerminal::Failed);
        drop(second_guard);
        current_command_profiling()
            .expect("first restored")
            .begin_provider_operation()
            .expect("first contribution")
            .complete(ProviderObservationTerminal::Missing);
        assert_eq!(
            first_receipt
                .seal()
                .observations
                .expect("first valid")
                .missing,
            1
        );
        assert_eq!(
            second_receipt
                .seal()
                .observations
                .expect("second valid")
                .failed,
            1
        );
    }

    #[test]
    fn scheduled_context_authorizes_only_its_exact_accelerator_lease() {
        use runmat_accelerate_api::AccelProvider as _;
        use runmat_execution::identity::{AttemptId, ExecutionScopeId, PoolId, TaskId, WorkerId};
        use runmat_execution::resource::{AcceleratorDeviceLease, ResourceAssignment};

        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let attempt_id = AttemptId::derive(&[b"provider-authorization"]);
        let device = LEASED_PROVIDER
            .execution_accelerator_device(7)
            .unwrap()
            .unwrap();
        let lease = AcceleratorDeviceLease::for_attempt(attempt_id, 11, &device).unwrap();
        let assignment = runmat_execution::ProgramExecutionAssignment {
            scope_id: ExecutionScopeId::derive(&[b"scope"]),
            pool_id: PoolId::derive(&[b"pool"]),
            task_id: TaskId::derive(&[b"task"]),
            attempt_id,
            worker_id: WorkerId::derive(&[b"worker"]),
            backend: runmat_execution::PoolBackend::LocalProcesses,
            resources: ResourceAssignment {
                accelerator_leases: vec![lease],
            },
        };

        let _context = runtime.enter();
        assert_eq!(
            execution_provider_authorization(
                runmat_accelerate_api::ExecutionProviderAdmission::Provider(&LEASED_PROVIDER)
            ),
            runmat_accelerate_api::ExecutionProviderAuthorization::Unrestricted
        );
        let _assignment = runtime.enter_execution_assignment(Some(assignment));
        assert_eq!(
            execution_provider_authorization(
                runmat_accelerate_api::ExecutionProviderAdmission::Provider(&LEASED_PROVIDER)
            ),
            runmat_accelerate_api::ExecutionProviderAuthorization::Allowed
        );
        assert_eq!(
            execution_provider_authorization(
                runmat_accelerate_api::ExecutionProviderAdmission::Provider(&OTHER_PROVIDER)
            ),
            runmat_accelerate_api::ExecutionProviderAuthorization::Denied
        );
    }

    #[test]
    fn synchronous_context_guard_restores_the_previous_context() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let first_scope = first.execution().scope_id();
        let second_scope = second.execution().scope_id();
        {
            let _first = first.enter();
            assert_eq!(
                active_runtime_context().unwrap().execution().scope_id(),
                first_scope
            );
            {
                let _second = second.enter();
                assert_eq!(
                    active_runtime_context().unwrap().execution().scope_id(),
                    second_scope
                );
            }
            assert_eq!(
                active_runtime_context().unwrap().execution().scope_id(),
                first_scope
            );
        }
        assert!(active_runtime_context().is_none());
    }

    #[test]
    fn execution_stack_guard_restores_nested_stack_state() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Process);
        {
            let _segmented = runtime.enter_execution_stack(RuntimeExecutionStack::Segmented);
            assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Segmented);
            {
                let _process = runtime.enter_execution_stack(RuntimeExecutionStack::Process);
                assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Process);
            }
            assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Segmented);
        }
        assert_eq!(runtime.execution_stack(), RuntimeExecutionStack::Process);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn interleaved_context_futures_restore_the_active_session() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let first_scope = first.execution().scope_id();
        let second_scope = second.execution().scope_id();

        let mut first_future = Box::pin(first.scope(std::future::poll_fn(move |_| {
            assert_eq!(
                active_runtime_context()
                    .expect("first context active")
                    .execution()
                    .scope_id(),
                first_scope
            );
            Poll::<()>::Pending
        })));
        let mut second_future = Box::pin(second.scope(std::future::poll_fn(move |_| {
            assert_eq!(
                active_runtime_context()
                    .expect("second context active")
                    .execution()
                    .scope_id(),
                second_scope
            );
            Poll::<()>::Pending
        })));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);

        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(active_runtime_context().is_none());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(active_runtime_context().is_none());
    }

    #[test]
    fn interleaved_futures_isolate_source_output_and_cancellation_state() {
        async fn session_future(
            source_id: SourceId,
            source: &'static str,
            outputs: usize,
            cancelled: bool,
        ) {
            let _catalog = crate::source_context::replace_source_catalog(vec![(
                source_id,
                format!("session-{}.m", source_id.0),
                source.to_string(),
            )])
            .expect("session source catalog admission");
            let _source = crate::source_context::replace_current_source_id(Some(source_id));
            let _outputs = crate::output_count::push_output_count(Some(outputs))
                .expect("session output-count projection");
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(
                crate::source_context::current_source().as_deref(),
                Some(source)
            );
            assert_eq!(crate::output_count::current_output_count(), Some(outputs));
            assert_eq!(crate::interrupt::is_cancelled(), cancelled);
        }

        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let first = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            cancellation.token(),
        );
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let mut first_future = Box::pin(first.scope(session_future(SourceId(1), "first", 1, true)));
        let mut second_future =
            Box::pin(second.scope(session_future(SourceId(2), "second", 2, false)));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);

        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        cancellation.cancel(runmat_execution::CancellationReason::User);
        assert!(first_future.as_mut().poll(&mut cx).is_ready());
        assert!(second_future.as_mut().poll(&mut cx).is_ready());
        assert!(crate::source_context::current_source().is_none());
        assert_eq!(crate::output_count::current_output_count(), None);
    }

    #[test]
    fn interleaved_futures_isolate_host_interaction_handlers() {
        async fn session_future(label: &'static str) {
            let handler: Arc<crate::interaction::AsyncInteractionHandler> = Arc::new(move |_| {
                Box::pin(async move {
                    Ok(crate::interaction::InteractionResponse::Line(
                        label.to_string(),
                    ))
                })
            });
            let _handler = crate::interaction::replace_async_handler(Some(handler));
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(
                crate::interaction::request_line_async("", false)
                    .await
                    .unwrap(),
                label
            );
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let mut first_future = Box::pin(first.scope(session_future("first")));
        let mut second_future = Box::pin(second.scope(session_future("second")));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(first_future.as_mut().poll(&mut cx).is_pending());
        assert!(second_future.as_mut().poll(&mut cx).is_pending());
        assert!(first_future.as_mut().poll(&mut cx).is_ready());
        assert!(second_future.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn dropping_scoped_future_activates_its_context_for_guard_cleanup() {
        struct DropProbe(runmat_execution::ExecutionScopeId);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                assert_eq!(
                    active_runtime_context()
                        .expect("context active while inner future drops")
                        .execution()
                        .scope_id(),
                    self.0
                );
            }
        }

        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let scope_id = runtime.execution().scope_id();
        let future = async move {
            let _probe = DropProbe(scope_id);
            std::future::pending::<()>().await;
        };
        let mut future = Box::pin(runtime.scope(future));
        let waker = std::task::Waker::from(Arc::new(NoopWake));
        let mut cx = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut cx).is_pending());
        drop(future);
        assert!(active_runtime_context().is_none());

        // The admitted inline owner has the same construction, poll, and
        // destruction semantics without a second future allocation. Explicit
        // absence masks foreign runtime and profiling owners, including before
        // the future has ever been polled.
        struct InlineProbe {
            expected: Option<runmat_execution::ExecutionScopeId>,
            dropped: Rc<std::cell::Cell<usize>>,
        }
        impl InlineProbe {
            fn check(&self) {
                assert_eq!(
                    active_runtime_context().map(|r| r.execution().scope_id()),
                    self.expected
                );
                assert!(current_command_profiling().is_none());
            }
        }
        impl Drop for InlineProbe {
            fn drop(&mut self) {
                self.check();
                self.dropped.set(self.dropped.get() + 1);
            }
        }
        let foreign_receipt = CommandProfilingRoot::new();
        let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_command_profiling(foreign_receipt.receipt());
        let _foreign = foreign.enter();
        for selected in [Some(runtime.clone()), None] {
            for poll in [false, true] {
                let dropped = Rc::new(std::cell::Cell::new(0));
                let probe = InlineProbe {
                    expected: selected.as_ref().map(|r| r.execution().scope_id()),
                    dropped: dropped.clone(),
                };
                let factory = RuntimeScopeFactory::new(selected.clone(), move || {
                    probe.check();
                    async move {
                        probe.check();
                        std::future::pending::<()>().await;
                        drop(probe);
                        Ok::<(), crate::RuntimeError>(())
                    }
                });
                let mut future = Box::pin(factory.construct());
                if poll {
                    assert!(future.as_mut().poll(&mut cx).is_pending());
                }
                drop(future);
                assert_eq!(dropped.get(), 1);
                assert_eq!(
                    active_runtime_context().unwrap().execution().scope_id(),
                    foreign.execution().scope_id()
                );
                assert!(current_command_profiling().is_some());

                let lazy_dropped = Rc::new(std::cell::Cell::new(0));
                let lazy_probe = InlineProbe {
                    expected: selected.as_ref().map(|r| r.execution().scope_id()),
                    dropped: lazy_dropped.clone(),
                };
                let mut lazy = Box::pin(LazyRuntimeScope::new(selected.clone(), async move {
                    lazy_probe.check();
                    std::future::pending::<()>().await;
                    drop(lazy_probe);
                    Ok::<(), crate::RuntimeError>(())
                }));
                if poll {
                    assert!(lazy.as_mut().poll(&mut cx).is_pending());
                }
                drop(lazy);
                assert_eq!(lazy_dropped.get(), 1);
                assert_eq!(
                    active_runtime_context().unwrap().execution().scope_id(),
                    foreign.execution().scope_id()
                );
                assert!(current_command_profiling().is_some());
            }
        }

        // A synchronously captured projection uses the same inline drop owner,
        // including an unpolled child and a genuinely pending child.
        for poll in [false, true] {
            let dropped = Rc::new(std::cell::Cell::new(0));
            let probe = InlineProbe {
                expected: Some(runtime.execution().scope_id()),
                dropped: dropped.clone(),
            };
            let scope = runtime.prepare_poll_scope().unwrap();
            let mut future = Box::pin(scope.construct_result_unboxed(move |captured| {
                assert_eq!(Some(captured.execution().scope_id()), probe.expected);
                probe.check();
                async move {
                    probe.check();
                    std::future::pending::<()>().await;
                    drop(probe);
                    Ok::<(), crate::RuntimeError>(())
                }
            }));
            if poll {
                assert!(future.as_mut().poll(&mut cx).is_pending());
            }
            drop(future);
            assert_eq!(dropped.get(), 1);
            assert_eq!(
                active_runtime_context().unwrap().execution().scope_id(),
                foreign.execution().scope_id()
            );
            assert!(current_command_profiling().is_some());
        }

        // Construction does not snapshot the ambient request. The selected
        // first-poll request survives its caller guard retiring between polls.
        let factory = RuntimeScopeFactory::new(Some(runtime.clone()), || async {
            assert_eq!(crate::output_count::current_output_count(), Some(3));
            let mut yielded = false;
            std::future::poll_fn(move |_| {
                if yielded {
                    Poll::Ready(())
                } else {
                    yielded = true;
                    Poll::Pending
                }
            })
            .await;
            assert_eq!(crate::output_count::current_output_count(), Some(3));
            Ok::<(), crate::RuntimeError>(())
        });
        let mut future = Box::pin(factory.construct());
        {
            let _selected = runtime.enter();
            let _count = crate::output_count::push_output_count(Some(3)).unwrap();
            assert!(future.as_mut().poll(&mut cx).is_pending());
        }
        assert!(matches!(future.as_mut().poll(&mut cx), Poll::Ready(Ok(()))));

        // Pending cleanup pins its creation token without replacing routes or
        // first-poll instruction projections. The caller's rebound slot is
        // restored after each synchronous construction/destruction boundary.
        struct TokenProbe {
            token: runmat_execution::CancellationToken,
            dropped: Rc<std::cell::Cell<usize>>,
        }
        impl Drop for TokenProbe {
            fn drop(&mut self) {
                let active = active_runtime_context().expect("original cleanup runtime");
                assert!(active.cancellation().same_generation(&self.token));
                self.dropped.set(self.dropped.get() + 1);
            }
        }
        let original_token = runtime.cancellation();
        let successor = runmat_execution::cancellation::CancellationSource::new();
        for construct in [false, true] {
            let dropped = Rc::new(std::cell::Cell::new(0));
            let probe = TokenProbe {
                token: original_token.clone(),
                dropped: dropped.clone(),
            };
            let factory = RuntimeScopeFactory::new(Some(runtime.clone()), move || async move {
                let _probe = probe;
                std::future::pending::<Result<(), crate::RuntimeError>>().await
            });
            if construct {
                let future = factory.construct();
                let _rebound = runtime.scoped_cancellation_token(successor.token());
                drop(future);
                assert!(runtime.cancellation().same_generation(&successor.token()));
            } else {
                let _rebound = runtime.scoped_cancellation_token(successor.token());
                drop(factory);
                assert!(runtime.cancellation().same_generation(&successor.token()));
            }
            assert_eq!(dropped.get(), 1);
        }
        for poll in [false, true] {
            let dropped = Rc::new(std::cell::Cell::new(0));
            let entered = Rc::new(std::cell::Cell::new(false));
            let probe = TokenProbe {
                token: original_token.clone(),
                dropped: dropped.clone(),
            };
            let child_entered = entered.clone();
            let mut lazy = Box::pin(LazyRuntimeScope::new(Some(runtime.clone()), async move {
                let _probe = probe;
                child_entered.set(true);
                std::future::pending::<Result<(), crate::RuntimeError>>().await
            }));
            let _rebound = runtime.scoped_cancellation_token(successor.token());
            if poll {
                assert!(matches!(lazy.as_mut().poll(&mut cx), Poll::Ready(Err(_))));
            }
            drop(lazy);
            assert!(!entered.get());
            assert_eq!(dropped.get(), 1);
            assert!(runtime.cancellation().same_generation(&successor.token()));
        }
        // Rebound-before-construction is a terminal rejection, not a delayed
        // retry after the mutable slot happens to return to the old token.
        let invoked = Rc::new(std::cell::Cell::new(false));
        let dropped = Rc::new(std::cell::Cell::new(0));
        let probe = TokenProbe {
            token: original_token.clone(),
            dropped: dropped.clone(),
        };
        let made = invoked.clone();
        let factory = RuntimeScopeFactory::new(Some(runtime.clone()), move || {
            made.set(true);
            async move {
                drop(probe);
                Ok::<(), crate::RuntimeError>(())
            }
        });
        let mut rejected = {
            let _rebound = runtime.scoped_cancellation_token(successor.token());
            let future = Box::pin(factory.construct());
            assert!(!invoked.get());
            assert_eq!(dropped.get(), 1);
            assert!(runtime.cancellation().same_generation(&successor.token()));
            future
        };
        assert!(runtime.cancellation().same_generation(&original_token));
        assert!(matches!(
            rejected.as_mut().poll(&mut cx),
            Poll::Ready(Err(_))
        ));
        assert!(!invoked.get());
        drop(rejected);
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn workspace_resolver_registration_is_context_scoped_in_production_shape() {
        fn first_lookup(name: &str) -> Option<runmat_value::Value> {
            (name == "x").then_some(runmat_value::Value::Num(1.0))
        }
        fn second_lookup(name: &str) -> Option<runmat_value::Value> {
            (name == "x").then_some(runmat_value::Value::Num(2.0))
        }
        fn empty_snapshot() -> Vec<(String, runmat_value::Value)> {
            Vec::new()
        }
        fn empty_globals() -> Vec<String> {
            Vec::new()
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
                lookup: first_lookup,
                snapshot: empty_snapshot,
                globals: empty_globals,
                assign: None,
                clear: None,
                remove: None,
            });
        }
        {
            let _scope = RuntimeContextGuard::enter(second.clone());
            crate::workspace::register_workspace_resolver(crate::workspace::WorkspaceResolver {
                lookup: second_lookup,
                snapshot: empty_snapshot,
                globals: empty_globals,
                assign: None,
                clear: None,
                remove: None,
            });
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert_eq!(
                crate::workspace::lookup("x"),
                Some(runmat_value::Value::Num(1.0))
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert_eq!(
                crate::workspace::lookup("x"),
                Some(runmat_value::Value::Num(2.0))
            );
        }
    }

    #[test]
    fn class_and_static_property_state_is_session_owned() {
        fn class(name: &str) -> crate::class_registry::RuntimeClass {
            crate::class_registry::RuntimeClass {
                events: Default::default(),
                name: name.into(),
                parent: None,
                properties: std::collections::HashMap::from([(
                    "answer".into(),
                    crate::class_registry::RuntimeProperty {
                        name: "answer".into(),
                        is_static: true,
                        is_constant: false,
                        is_dependent: false,
                        get_access: runmat_types::MemberAccess::Public,
                        set_access: runmat_types::MemberAccess::Public,
                        default_value: None,
                    },
                )]),
                methods: std::collections::HashMap::new(),
            }
        }

        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let only_first = runmat_types::ClassIdentity::from("OnlyFirst");
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::class_registry::register_class(class("OnlyFirst"));
            crate::class_registry::set_static_property_value(
                &only_first,
                "answer",
                runmat_value::Value::Num(1.0),
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert!(crate::class_registry::get_class(&only_first).is_none());
            assert!(
                crate::class_registry::get_static_property_value(&only_first, "answer").is_none()
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert!(crate::class_registry::get_class(&only_first).is_some());
            assert_eq!(
                crate::class_registry::get_static_property_value(&only_first, "answer"),
                Some(runmat_value::Value::Num(1.0))
            );
        }
    }

    #[test]
    fn lazy_builtin_class_registration_is_session_owned() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        {
            let _scope = RuntimeContextGuard::enter(first.clone());
            crate::testing::ensure_testing_classes();
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_some()
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(second.clone());
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_none()
            );
            crate::testing::ensure_testing_classes();
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_some()
            );
        }
        {
            let _scope = RuntimeContextGuard::enter(first);
            assert!(
                crate::class_registry::get_class(&crate::testing::TEST_CASE_CLASS.owned())
                    .is_some()
            );
        }
    }

    #[test]
    fn error_and_compatibility_policy_is_session_owned() {
        let first = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let second = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        first.set_error_namespace("First");
        first.set_callstack_limit(11);
        second.set_error_namespace("Second");
        second.set_callstack_limit(22);
        {
            let _scope = RuntimeContextGuard::enter(first);
            crate::compatibility::set_runmat_extensions_enabled(true);
            assert_eq!(active_runtime_context().unwrap().error_namespace(), "First");
            assert_eq!(active_runtime_context().unwrap().callstack_limit(), 11);
            assert!(crate::compatibility::runmat_extensions_enabled());
        }
        {
            let _scope = RuntimeContextGuard::enter(second);
            assert_eq!(
                active_runtime_context().unwrap().error_namespace(),
                "Second"
            );
            assert_eq!(active_runtime_context().unwrap().callstack_limit(), 22);
            assert!(!crate::compatibility::runmat_extensions_enabled());
        }
    }
}
