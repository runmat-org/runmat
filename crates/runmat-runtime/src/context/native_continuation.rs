//! Foreground native calls stay direct; a native child of a native invocation
//! is submitted to the host's already-driven local task supervisor.

use std::{future::Future, rc::Rc};

use futures::{channel::oneshot, future::Either};

use super::{CanonicalLocalTaskAdmission, RuntimeContext};
use crate::{
    call::RootedFunctionArguments,
    execution::{RootedValueLease, RuntimeContinuationFuture},
    RuntimeError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeContinuationPlacement {
    sibling: bool,
}

impl NativeContinuationPlacement {
    pub fn is_direct(self) -> bool {
        !self.sibling
    }
}

impl RuntimeContext {
    /// Capture the ancestry before installing callback guards. The returned
    /// context is propagated into every child invocation and selected service.
    pub fn native_invocation_context(&self) -> (Self, NativeContinuationPlacement) {
        let placement = NativeContinuationPlacement {
            sibling: self.native_invocation_lineage.is_some(),
        };
        let mut selected = self.clone();
        selected.native_invocation_lineage = Some(
            self.native_invocation_lineage
                .as_ref()
                .cloned()
                .unwrap_or_else(|| Rc::new(())),
        );
        (selected, placement)
    }
}

struct CancelOnDrop(super::RuntimeLocalTaskLease);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Keep semantic catch authority inside the original admitted concrete body,
/// before it is erased or queued. The quote includes this unboxed wrapper,
/// and only the original Rc is retained, not a copied exception payload.
fn admit_catch_scoped_native_continuation_with<T, F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<RuntimeContinuationFuture<'static, T>, RuntimeError>
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    let selected = runtime.clone();
    let catch = runtime.active_catch_exception_shared();
    crate::execution::rooted_continuation::admit_native_continuation_with(runtime, move || {
        selected.scope_active_catch_unboxed_shared(catch, make_future())
    })
}

/// Retain already-owned inputs until the child finishes or its ticket is
/// cancelled. The helper does not copy a workspace or open a second driver.
pub fn run_native_continuation<T, F>(
    runtime: &RuntimeContext,
    placement: NativeContinuationPlacement,
    roots: RootedValueLease,
    future: F,
) -> impl Future<Output = Result<T, RuntimeError>> + 'static
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    run_native_continuation_with(runtime, placement, roots, || future)
}

/// Factory form for native executors whose concrete future owns a large
/// interpreter state. Admission happens before the factory runs, and callers
/// retain the same direct-versus-sibling placement semantics.
pub fn run_native_continuation_with<T, F>(
    runtime: &RuntimeContext,
    placement: NativeContinuationPlacement,
    roots: RootedValueLease,
    make_future: impl FnOnce() -> F + 'static,
) -> impl Future<Output = Result<T, RuntimeError>> + 'static
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    let selected = runtime.clone();
    let admitted = admit_catch_scoped_native_continuation_with(runtime, make_future);
    match admitted {
        Ok(future) => Either::Left(async move {
            run_native_continuation_with_roots(&selected, placement, Some(roots), future).await
        }),
        Err(error) => Either::Right(std::future::ready(Err(error))),
    }
}

/// A selected callable already carries its capture generation and supplied
/// input roots. Move that exact owner into the child future before queueing;
/// no second traversal, registration, or empty proof lease is fabricated.
pub fn run_rooted_function_native_continuation<T, F>(
    runtime: &RuntimeContext,
    placement: NativeContinuationPlacement,
    arguments: RootedFunctionArguments,
    consume: impl FnOnce(RootedFunctionArguments) -> F + 'static,
) -> impl Future<Output = Result<T, RuntimeError>> + 'static
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    let selected = runtime.clone();
    let admitted = admit_catch_scoped_native_continuation_with(runtime, move || async move {
        consume(arguments).await
    });
    match admitted {
        Ok(future) => Either::Left(async move {
            run_native_continuation_with_roots(&selected, placement, None, future).await
        }),
        Err(error) => Either::Right(std::future::ready(Err(error))),
    }
}

async fn run_native_continuation_with_roots<T>(
    runtime: &RuntimeContext,
    placement: NativeContinuationPlacement,
    roots: Option<RootedValueLease>,
    future: RuntimeContinuationFuture<'static, T>,
) -> Result<T, RuntimeError>
where
    T: 'static,
{
    if placement.is_direct() {
        let _roots = roots;
        return runtime.scope(future).await;
    }
    let tasks = runtime
        .service_ports()
        .require_local_tasks("continue nested native invocation")
        .map_err(super::RuntimeCapabilityError::into_runtime_error)?;
    let admission = tasks.canonical_admission().ok_or_else(|| {
        crate::build_runtime_error("nested native invocation needs a canonical local-task host")
            .with_identifier("RunMat:RuntimeContext:NativeContinuationHost")
            .build()
    })?;
    spawn_foreground_sibling(runtime, admission, roots, future).await
}

/// An already-owned callback may outlive its caller's poll. When the selected
/// host has a canonical local-task driver, poll it as a sibling so a nested VM
/// or native callback cannot consume the suspended caller's stack. Bare hosts
/// retain their existing direct completion path. A configured but idle task
/// host is also direct: queueing there would wait for a driver that the caller
/// is not running.
pub fn run_selected_callback_continuation<'a, T, F>(
    runtime: &'a RuntimeContext,
    future: F,
) -> impl Future<Output = Result<T, RuntimeError>> + 'a
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    // A selected semantic callback can be polled as a sibling after its
    // caller's lexical catch has unwound. Keep that exact Rc (or an explicit
    // mask) inside the original concrete callback owner, including on drop.
    let future =
        runtime.scope_active_catch_unboxed_shared(runtime.active_catch_exception_shared(), future);
    run_catch_scoped_selected_callback_continuation(runtime, future)
}

async fn run_catch_scoped_selected_callback_continuation<T, F>(
    runtime: &RuntimeContext,
    future: F,
) -> Result<T, RuntimeError>
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    let Some(admission) = runtime
        .service_ports()
        .local_tasks()
        .and_then(|tasks| tasks.canonical_admission())
    else {
        return runtime.scope(future).await;
    };
    if !admission.is_active_driver() {
        return runtime.scope(future).await;
    }
    spawn_foreground_sibling(runtime, admission, None, future).await
}

/// Factory form for selected callbacks whose concrete future contains a
/// substantial interpreter or foreign-service state. Admit and erase that
/// exact state before constructing it, then retain the ordinary direct versus
/// supervised-sibling callback placement.
#[inline(never)]
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run_selected_callback_continuation_with<T, F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F + 'static,
) -> impl Future<Output = Result<T, RuntimeError>> + 'static
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    let selected = runtime.clone();
    match admit_catch_scoped_native_continuation_with(runtime, make_future) {
        Ok(future) => Either::Left(async move {
            run_catch_scoped_selected_callback_continuation(&selected, future).await
        }),
        Err(error) => Either::Right(std::future::ready(Err(error))),
    }
}

/// Split an owned request's poll stack only while its exact host supervisor is
/// already driving. A caller awaiting directly outside that drive keeps its
/// original direct execution path instead of submitting to an idle LocalSet.
pub async fn run_owned_request_continuation<T, F>(
    runtime: &RuntimeContext,
    future: F,
) -> Result<T, RuntimeError>
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        let _ = runtime;
        future.await
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(admission) = runtime
            .service_ports()
            .local_tasks()
            .and_then(|tasks| tasks.canonical_admission())
        else {
            return future.await;
        };
        if !admission.is_active_driver() {
            return future.await;
        }
        spawn_foreground_sibling(runtime, admission, None, future).await
    }
}

async fn spawn_foreground_sibling<T, F>(
    runtime: &RuntimeContext,
    admission: CanonicalLocalTaskAdmission,
    roots: Option<RootedValueLease>,
    future: F,
) -> Result<T, RuntimeError>
where
    T: 'static,
    F: Future<Output = Result<T, RuntimeError>> + 'static,
{
    let (send, receive) = oneshot::channel();
    // Capture the selected call/source overlays while the parent poll is
    // active, before the sibling is queued and that poll unwinds.
    let selected = runtime.clone().scope(future);
    let child = Box::pin(async move {
        let _roots = roots;
        let result = selected.await;
        let _ = send.send(result);
    });
    let ticket = admission.spawn_admitted(runtime.local_identity(), child, None)?;
    let _cancel = CancelOnDrop(ticket);
    receive.await.map_err(|_| {
        crate::build_runtime_error("nested native invocation ended before publishing its result")
            .with_identifier("RunMat:RuntimeContext:NativeContinuationCancelled")
            .build()
    })?
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{
        context::{
            host_effect::ManagedHostEffectLedger, ParallelCapability, RuntimeParallelResources,
            RuntimeParallelService, RuntimeServicePorts,
        },
        execution::RuntimeExecutionService,
    };
    use runmat_value::Value;
    use std::cell::Cell;

    fn roots() -> RootedValueLease {
        RootedValueLease::for_values(std::iter::empty::<&Value>()).expect("empty roots")
    }

    fn host() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test reactor")
    }

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn direct_invocation_needs_no_task_service() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let (selected, placement) = runtime.native_invocation_context();
        assert!(placement.is_direct());
        let result = host().block_on(run_native_continuation(
            &selected,
            placement,
            roots(),
            async { Ok::<_, RuntimeError>(7) },
        ));
        assert_eq!(result.expect("direct result"), 7);
    }

    #[test]
    fn direct_factory_is_not_constructed_before_managed_admission() {
        let ledger = ManagedHostEffectLedger::new(1);
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(
                RuntimeServicePorts::default()
                    .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
            );
        let (selected, placement) = runtime.native_invocation_context();
        let constructed = Rc::new(Cell::new(false));
        let observed = Rc::clone(&constructed);
        let completion = run_native_continuation_with(&selected, placement, roots(), move || {
            observed.set(true);
            async { Ok::<_, RuntimeError>(7) }
        });
        assert!(!constructed.get());
        assert!(host().block_on(completion).is_err());
        assert!(!constructed.get());
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    }

    #[test]
    fn unpolled_rooted_function_does_not_consume_arguments() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let (selected, placement) = runtime.native_invocation_context();
        let arguments = RootedFunctionArguments::ordinary(
            crate::execution::RootedValues::new(Vec::new()).expect("empty rooted arguments"),
        );
        let consumed = Rc::new(Cell::new(false));
        let observed = Rc::clone(&consumed);
        let completion =
            run_rooted_function_native_continuation(&selected, placement, arguments, move |_| {
                observed.set(true);
                async { Ok::<_, RuntimeError>(()) }
            });
        assert!(!consumed.get());
        drop(completion);
        assert!(!consumed.get());
    }

    #[test]
    fn callback_preserves_direct_completion_without_an_active_task_driver() {
        let supervisor = super::super::RuntimeLocalTaskSupervisor::new();
        let bare = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let ports = RuntimeServicePorts::default()
            .with_local_tasks(supervisor.submission_port(bare.local_identity()));
        let idle = bare.clone().with_service_ports(ports);
        for runtime in [bare, idle] {
            let result = host().block_on(async {
                let mut completion =
                    std::pin::pin!(run_selected_callback_continuation(&runtime, async {
                        Ok::<_, RuntimeError>(5)
                    },));
                futures::future::poll_fn(|cx| std::task::Poll::Ready(completion.as_mut().poll(cx)))
                    .await
            });
            assert!(matches!(result, std::task::Poll::Ready(Ok(5))));
        }
    }

    #[test]
    fn nested_invocation_is_queued_under_the_existing_driver() {
        let supervisor = super::super::RuntimeLocalTaskSupervisor::new();
        let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let ports = RuntimeServicePorts::default()
            .with_local_tasks(supervisor.submission_port(base.local_identity()));
        let base = base.with_service_ports(ports);
        let (parent, first) = base.native_invocation_context();
        assert!(first.is_direct());
        let (child, second) = parent.native_invocation_context();
        assert!(!second.is_direct());
        let polls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&polls);
        let result = host()
            .block_on(supervisor.drive_until(async {
                let exception = Rc::new(runmat_value::MException::new(
                    "RunMat:nativeCaller".into(),
                    "native caller".into(),
                ));
                let guard = child.enter_active_catch_shared(Some(exception.clone()));
                let callback_runtime = child.clone();
                let completion = run_native_continuation(&child, second, roots(), async move {
                    assert!(Rc::ptr_eq(
                        &callback_runtime
                            .active_catch_exception_shared()
                            .expect("inherit native caller catch"),
                        &exception,
                    ));
                    observed.set(observed.get() + 1);
                    Ok::<_, RuntimeError>(11)
                });
                drop(guard);
                let result = completion.await;
                assert!(child.active_catch_exception_shared().is_none());
                assert_eq!(polls.get(), 1);
                result
            }))
            .expect("existing driver");
        assert_eq!(result.expect("child publication"), 11);
    }

    #[test]
    fn hosted_callback_does_not_poll_on_the_parent_stack() {
        let supervisor = super::super::RuntimeLocalTaskSupervisor::new();
        let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let ports = RuntimeServicePorts::default()
            .with_local_tasks(supervisor.submission_port(base.local_identity()));
        let base = base.with_service_ports(ports);
        let polls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&polls);
        let result = host()
            .block_on(supervisor.drive_until(async {
                let exception = Rc::new(runmat_value::MException::new(
                    "RunMat:callbackCaller".into(),
                    "callback caller".into(),
                ));
                let guard = base.enter_active_catch_shared(Some(exception.clone()));
                let callback_runtime = base.clone();
                let mut completion = Box::pin(run_selected_callback_continuation_with(
                    &base,
                    move || async move {
                        assert!(Rc::ptr_eq(
                            &callback_runtime
                                .active_catch_exception_shared()
                                .expect("inherit selected callback catch"),
                            &exception,
                        ));
                        observed.set(observed.get() + 1);
                        Ok::<_, RuntimeError>(17)
                    },
                ));
                drop(guard);
                let first_poll = futures::future::poll_fn(|cx| {
                    std::task::Poll::Ready(completion.as_mut().poll(cx))
                })
                .await;
                assert!(first_poll.is_pending());
                assert_eq!(polls.get(), 0, "callback must run after the parent yields");
                completion.await
            }))
            .expect("existing driver");
        assert_eq!(result.expect("suspended callback publication"), 17);
        assert_eq!(polls.get(), 1);

        let request_polls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&request_polls);
        let request = host()
            .block_on(supervisor.drive_until(async {
                let mut completion = Box::pin(run_owned_request_continuation(&base, async move {
                    observed.set(observed.get() + 1);
                    Ok::<_, RuntimeError>(19)
                }));
                let first_poll = futures::future::poll_fn(|cx| {
                    std::task::Poll::Ready(completion.as_mut().poll(cx))
                })
                .await;
                assert!(first_poll.is_pending());
                assert_eq!(
                    request_polls.get(),
                    0,
                    "owned request must yield to its host task"
                );
                completion.await
            }))
            .expect("existing driver");
        assert_eq!(request.expect("owned request publication"), 19);
        assert_eq!(request_polls.get(), 1);

        let direct = host().block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                run_owned_request_continuation(&base, async { Ok::<_, RuntimeError>(23) }),
            )
            .await
        });
        assert_eq!(
            direct.expect("idle host must not strand request").unwrap(),
            23
        );
    }

    #[test]
    fn nested_child_retains_the_exact_parent_call_projection() {
        let supervisor = super::super::RuntimeLocalTaskSupervisor::new();
        let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let ports = RuntimeServicePorts::default()
            .with_local_tasks(supervisor.submission_port(base.local_identity()));
        let base = base.with_service_ports(ports);
        let expected: Rc<crate::user_functions::BoundFunctionInvoker> =
            Rc::new(|runtime, _, _, _, _| {
                crate::execution::admit_executable_sequence_with(&runtime, move || async move {
                    crate::execution::RootedValueSequence::new(runmat_value::ValueSequence::empty())
                        .map_err(|error| error.into_runtime_error())
                })
            });
        let parent = base.clone();
        let selected = Rc::clone(&expected);
        let result = host()
            .block_on(supervisor.drive_until(base.scope(async move {
                let _guard = crate::user_functions::install_local_bound_function_invoker(selected)
                    .expect("call-port scope");
                let (parent, _) = parent.native_invocation_context();
                let (child, placement) = parent.native_invocation_context();
                run_native_continuation(&child, placement, roots(), async move {
                    let actual = crate::user_functions::current_bound_function_invoker()
                        .expect("selected bound invoker survived sibling scheduling");
                    assert!(Rc::ptr_eq(&actual, &expected));
                    Ok::<_, RuntimeError>(())
                })
                .await
            })))
            .expect("existing driver");
        result.expect("selected sibling result");
    }

    #[test]
    fn dropping_nested_waiter_cancels_its_admitted_child() {
        use std::{future::Future, pin::Pin, task::Context};

        struct DropWitness(Rc<Cell<bool>>);
        impl Drop for DropWitness {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let supervisor = super::super::RuntimeLocalTaskSupervisor::new();
        let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let ports = RuntimeServicePorts::default()
            .with_local_tasks(supervisor.submission_port(base.local_identity()));
        let base = base.with_service_ports(ports);
        let (parent, _) = base.native_invocation_context();
        let (child, placement) = parent.native_invocation_context();
        let dropped = Rc::new(Cell::new(false));
        let witness = DropWitness(Rc::clone(&dropped));
        let mut waiting = Box::pin(run_native_continuation(
            &child,
            placement,
            roots(),
            async move {
                let _witness = witness;
                std::future::pending::<Result<(), RuntimeError>>().await
            },
        ));
        let wake = futures::task::noop_waker();
        let mut context = Context::from_waker(&wake);
        assert!(Pin::as_mut(&mut waiting).poll(&mut context).is_pending());
        assert!(!dropped.get());
        drop(waiting);
        host()
            .block_on(supervisor.drive_until(async {}))
            .expect("cancelled child drain");
        assert!(dropped.get());
    }
}
