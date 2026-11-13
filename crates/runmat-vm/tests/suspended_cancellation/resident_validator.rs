//! A supported arguments-block validator suspends inside its provider download.
//! The ordinary builtin-dispatch replacement is intentionally not involved.

use super::support;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned, HostTensorView,
    ProviderPrecision, ThreadProviderGuard,
};
use runmat_runtime::builtin::{
    RuntimeBuiltinBinding, RuntimeBuiltinFuture, RuntimeBuiltinInvocation,
};
use runmat_runtime::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use runmat_value::{Value, ValueSequence};
use std::{
    cell::RefCell,
    future::Future,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex, OnceLock,
    },
    task::{Context, Poll},
};

struct ValidatorProvider {
    owner: TestAllocationOwner,
    receiver: Mutex<Option<futures::channel::oneshot::Receiver<bool>>>,
    downloads: AtomicUsize,
}

static PROVIDER: OnceLock<ValidatorProvider> = OnceLock::new();

impl AccelProvider for ValidatorProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "suspended validator download fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("validator fixture inputs are produced by their allocation owner")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("validator must use the native numeric download") })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        assert_eq!(handle.device_id(), self.device_id());
        handle
            .allocation_lease()
            .validate_live()
            .expect("live validator input");
        self.downloads.fetch_add(1, Ordering::SeqCst);
        let receiver = self
            .receiver
            .lock()
            .unwrap()
            .take()
            .expect("one validator download");
        Box::pin(async move {
            if !receiver.await? {
                anyhow::bail!("injected validator download failure");
            }
            handle.allocation_lease().validate_live()?;
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0]),
                shape: vec![1],
                storage: GpuTensorStorage::Real,
            })
        })
    }
}

struct State {
    provider: &'static ValidatorProvider,
    probe: Option<RetirementProbe>,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

struct InstalledState;
impl Drop for InstalledState {
    fn drop(&mut self) {
        STATE.with(|state| state.borrow_mut().take());
    }
}

fn produce(_: RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        let state = STATE.with(|slot| slot.borrow().as_ref().expect("fixture installed").clone());
        let provider = state.borrow().provider;
        let allocation = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0]),
                shape: vec![1],
                storage: GpuTensorStorage::Real,
            })
            .expect("valid resident validator fixture allocation");
        state.borrow_mut().probe = Some(allocation.probe);
        runmat_runtime::execution::RootedValueSequence::new(
            ValueSequence::single(Value::GpuTensor(allocation.handle))
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?,
        )
        .map_err(|error| error.into_runtime_error())
    })
}

struct Builtins;
impl RuntimeBuiltinService for Builtins {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        if name != "sin" {
            return vec![];
        }
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        vec![RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            produce,
        )]
    }
}

#[derive(Clone, Copy, Debug)]
enum Terminal {
    Drop,
    Cancel,
    Error,
    Success,
}

#[test]
fn arguments_validator_retains_resident_input_across_real_download_suspension() {
    for terminal in [
        Terminal::Drop,
        Terminal::Cancel,
        Terminal::Error,
        Terminal::Success,
    ] {
        runmat_gc::gc_test_context(|| run_case(terminal));
    }
}

fn run_case(terminal: Terminal) {
    runmat_vm::reset_thread_state_for_tests();
    let (sender, receiver) = futures::channel::oneshot::channel();
    let provider = PROVIDER.get_or_init(|| ValidatorProvider {
        owner: TestAllocationOwner::new(),
        receiver: Mutex::new(None),
        downloads: AtomicUsize::new(0),
    });
    {
        let mut pending = provider.receiver.lock().unwrap();
        assert!(
            pending.is_none(),
            "previous validator download must be finished or abandoned"
        );
        *pending = Some(receiver);
    }
    provider.downloads.store(0, Ordering::SeqCst);
    let _provider = ThreadProviderGuard::set(Some(provider));
    let state = Rc::new(RefCell::new(State {
        provider,
        probe: None,
    }));
    STATE.with(|slot| *slot.borrow_mut() = Some(state.clone()));
    let _installed = InstalledState;
    let source_text = "function y = checked(v); arguments; v {mustBeFinite}; end; y = v; end; x = checked(sin(0));";
    let bytecode = support::compile_source(source_text).expect("supported validator source");
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    )
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Builtins)));
    let vars = vec![];
    let mut execution = Box::pin(runmat_vm::interpret_workspace(
        &bytecode,
        runmat_vm::InterpreterVars::from_owned(vars),
        None,
        runtime,
        None,
        None,
    ));
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    assert!(execution
        .as_mut()
        .poll(&mut context)
        .map(|execution| execution.outcome)
        .is_pending());
    let probe = state.borrow().probe.as_ref().expect("producer ran").clone();
    assert_eq!(
        provider.downloads.load(Ordering::SeqCst),
        1,
        "validator suspended in native download"
    );
    assert_eq!(
        probe.retirements(),
        0,
        "validator must retain its resident argument"
    );

    let mut published = None;
    match terminal {
        Terminal::Drop => {}
        Terminal::Cancel => {
            cancellation.cancel(runmat_execution::CancellationReason::User);
            let Poll::Ready(Err(error)) = execution
                .as_mut()
                .poll(&mut context)
                .map(|execution| execution.outcome)
            else {
                panic!("cancellation must terminalize the validator");
            };
            assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
        }
        Terminal::Error | Terminal::Success => {
            sender.send(matches!(terminal, Terminal::Success)).unwrap();
            let Poll::Ready(result) = execution
                .as_mut()
                .poll(&mut context)
                .map(|execution| execution.outcome)
            else {
                panic!("validator must finish after download result");
            };
            if matches!(terminal, Terminal::Success) {
                let runmat_vm::InterpreterOutcome::Completed(completion) = result.unwrap();
                let slot = bytecode
                    .var_names
                    .iter()
                    .find_map(|(slot, name)| (name == "x").then_some(*slot))
                    .unwrap();
                let handle = {
                    let value = completion
                        .values
                        .get(slot)
                        .expect("completed resident value remains in its compiled slot");
                    let Value::GpuTensor(handle) = &*value else {
                        panic!("successful validator must return the resident argument");
                    };
                    handle.clone()
                };
                published = Some(handle);
            } else {
                assert!(result.is_err(), "download failure must remain an error");
            }
        }
    }
    drop(execution);

    runmat_vm::reset_thread_state_for_tests();
    assert_eq!(
        provider.downloads.load(Ordering::SeqCst),
        1,
        "no validator replay"
    );
    assert_eq!(
        probe.retirements(),
        usize::from(published.is_none()),
        "{terminal:?} release boundary"
    );
    drop(published);
    assert_eq!(
        probe.retirements(),
        1,
        "final resident owner retired exactly once"
    );
}
