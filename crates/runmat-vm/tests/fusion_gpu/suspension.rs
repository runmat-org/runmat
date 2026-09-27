//! Compiler-produced typed fusion must wake without provider cooperation.
use super::*;
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct Probe {
    polls: Cell<usize>,
    drops: Cell<usize>,
}

thread_local! {
    static PENDING: RefCell<Option<Rc<Probe>>> = const { RefCell::new(None) };
    static DOWNLOAD: RefCell<Option<(u64, Rc<Probe>)>> = const { RefCell::new(None) };
}

struct Scope;

fn publish_borrowed(
    runtime: &runmat_runtime::context::RuntimeContext,
    handles: &[&GpuTensorHandle],
) {
    // A raw handle clone is not an independent owner. Publish through the real
    // session authority retained by the fixture across execution cancellation.
    block_on(runtime.scope(async {
        for (index, handle) in handles.iter().enumerate() {
            runmat_runtime::workspace::session::store_global_named(
                &format!("suspension_borrowed_{index}"),
                Value::GpuTensor((*handle).clone()),
            );
        }
    }));
}

fn unpublish_borrowed(runtime: &runmat_runtime::context::RuntimeContext) {
    block_on(runtime.scope(async {
        for name in runmat_runtime::workspace::session::global_names() {
            if name.starts_with("suspension_borrowed_") {
                runmat_runtime::workspace::session::store_global_named(&name, Value::Num(0.0));
            }
        }
    }));
}

fn assert_published_buffers(
    runtime: &runmat_runtime::context::RuntimeContext,
    provider: &TestProvider,
) {
    block_on(runtime.scope(async {
        for value in runmat_runtime::workspace::session::roots() {
            let Value::GpuTensor(handle) = value else {
                panic!("fixture publishes only borrowed device values")
            };
            assert_eq!(provider.pull(&handle).unwrap().1, handle.shape);
        }
    }));
}
impl Drop for Scope {
    fn drop(&mut self) {
        PENDING.with(|slot| slot.replace(None));
        DOWNLOAD.with(|slot| slot.replace(None));
    }
}

struct PendingDrop(Rc<Probe>);
impl Drop for PendingDrop {
    fn drop(&mut self) {
        assert!(runmat_runtime::context::legacy::active().is_some());
        self.0.drops.set(self.0.drops.get() + 1);
    }
}

pub(super) async fn pending_single_if_armed() {
    let probe = PENDING.with(|slot| slot.borrow().clone());
    if let Some(probe) = probe {
        assert_eq!(
            runmat_accelerate::fusion::active_group_plan_clone()
                .expect("Single restoration must run inside the production fusion region")
                .group
                .kind,
            FusionKind::ElementwiseChain,
        );
        probe.polls.set(probe.polls.get() + 1);
        let _drop = PendingDrop(probe);
        futures::future::pending::<()>().await;
    }
}

pub(super) async fn pending_download_if_armed(handle: &GpuTensorHandle) {
    let probe = DOWNLOAD.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|(id, _)| *id == handle.buffer_id)
            .map(|(_, probe)| probe.clone())
    });
    if let Some(probe) = probe {
        probe.polls.set(probe.polls.get() + 1);
        let _drop = PendingDrop(probe);
        futures::future::pending::<()>().await;
    }
}

#[test]
fn pending_stochastic_control_releases_only_owned_state() {
    for resident in [false, true] {
        for cancel in [false, true] {
            stochastic_witness(resident, cancel);
        }
    }
}

fn stochastic_witness(resident: bool, cancel: bool) {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        let bytecode = compile_semantic("M = 2; S = [1;2]; drift = 0.1; scale = 0.2; for t = 1:3; Z = randn(M,1); S = S .* exp(drift + scale .* Z); end");
        assert!(bytecode
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instr::StochasticEvolution)));
        let slot = |name: &str| {
            *bytecode
                .var_names
                .iter()
                .find(|(_, candidate)| candidate.as_str() == name)
                .unwrap()
                .0
        };
        let scale_slot = slot("scale");
        let store = bytecode
            .instructions
            .iter()
            .position(
                |instruction| matches!(instruction, Instr::StoreVar(index) if *index == scale_slot),
            )
            .unwrap();
        // Resume only after the compiler's literal input construction, without
        // editing the idiom instruction or injecting a synthetic suspension.
        assert!(bytecode.instructions[..store]
            .iter()
            .all(|instruction| matches!(
                instruction,
                Instr::LoadConst(_)
                    | Instr::LoadInt(_)
                    | Instr::StoreVar(_)
                    | Instr::CreateMatrix(_, _)
                    | Instr::CreateMatrixDynamic(_)
                    | Instr::EnterScope(_)
            )));
        let drift = provider.push(vec![0.1], vec![1, 1]);
        let borrowed = resident.then(|| provider.push(vec![1., 2.], vec![2, 1]));
        let baseline = provider.buffers.lock().unwrap().clone();
        let mut vars = vec![None; bytecode.var_count];
        vars[slot("M")] = Some(Value::Num(2.));
        vars[slot("S")] = Some(borrowed.as_ref().map_or_else(
            || Value::Tensor(Tensor::new(vec![1., 2.], vec![2, 1]).unwrap()),
            |handle| Value::GpuTensor(handle.clone()),
        ));
        vars[slot("drift")] = Some(Value::GpuTensor(drift.clone()));
        vars[scale_slot] = Some(Value::Num(0.2));
        let resume = runmat_vm::InterpreterResumeState {
            pc: store + 1,
            completion_boundary: None,
            vars,
            supplied_inputs: 0,
            requested_outputs: 0,
            missing_input_slots: Default::default(),
            global_aliases: Default::default(),
            persistent_aliases: Default::default(),
            side_effect_epoch: 0,
        };
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = runmat_runtime::context::RuntimeContext::with_cancellation(
            Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let mut published = vec![&drift];
        published.extend(borrowed.as_ref());
        publish_borrowed(&runtime, &published);
        let probe = Rc::new(Probe::default());
        assert!(DOWNLOAD
            .with(|slot| slot.replace(Some((drift.buffer_id, probe.clone()))))
            .is_none());
        let _scope = Scope;
        let mut execution = Box::pin(runmat_vm::interpret_resume_in_context(
            &bytecode,
            resume,
            None,
            runtime.clone(),
        ));
        let wakes = Arc::new(Wakes::default());
        let waker = Waker::from(wakes.clone());
        let mut cx = Context::from_waker(&waker);
        assert!(execution.as_mut().poll(&mut cx).is_pending());
        assert_eq!(probe.polls.get(), 1);
        assert_eq!(
            provider.buffers.lock().unwrap().len(),
            baseline.len() + usize::from(!resident),
            "state upload must precede pending drift download"
        );
        assert_published_buffers(&runtime, provider);
        if cancel {
            source.cancel(runmat_execution::CancellationReason::User);
            assert!(wakes.0.load(Ordering::SeqCst) > 0);
            let Poll::Ready(Err(error)) = execution.as_mut().poll(&mut cx) else {
                panic!("pending idiom cancellation");
            };
            assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
            assert_eq!(probe.drops.get(), 1);
        }
        drop(execution);
        assert_eq!(probe.drops.get(), 1);
        assert_eq!(probe.polls.get(), 1);
        assert_eq!(*provider.buffers.lock().unwrap(), baseline);
        assert_published_buffers(&runtime, provider);
        unpublish_borrowed(&runtime);
        provider.free(&drift).unwrap();
        if let Some(handle) = borrowed {
            provider.free(&handle).unwrap();
        }
        assert!(runmat_runtime::context::legacy::active().is_none());
    });
}

#[derive(Default)]
struct Wakes(AtomicU64);
impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn pending_typed_fusion_cancellation_preserves_borrowed_input() {
    witness(true, true);
    witness(true, false);
}

#[test]
fn pending_typed_fusion_cancellation_releases_uploaded_input() {
    // This also exercises the shared fusion allocation owner: the cancellation
    // boundary alone cannot release raw handles allocated by a compound kernel.
    witness(false, true);
    witness(false, false);
}

fn witness(resident: bool, cancel: bool) {
    gc_test_context(|| {
        let _lock = accel_test_lock();
        ensure_provider_registered();
        let provider = PROVIDER.get().unwrap();
        // The semantic compiler admits elementwise windows, not CenteredGram
        // windows. Single output restoration is an actual awaited provider
        // boundary after the fused F64 submission and owned allocations.
        let source = "x = 1:4096; y = single(sin(x) .* x + 2);";
        let bytecode = compile_semantic(source);
        assert!(bytecode
            .runtime_fusion_groups()
            .iter()
            .any(|group| { group.kind == FusionKind::ElementwiseChain }));
        let data = (1..=4096).map(f64::from).collect::<Vec<_>>();
        let borrowed = resident.then(|| provider.push(data.clone(), vec![1, 4096]));
        let input = borrowed.as_ref().map_or_else(
            || Value::Tensor(Tensor::new(data, vec![1, 4096]).unwrap()),
            |handle| Value::GpuTensor(handle.clone()),
        );
        let baseline = provider.buffers.lock().unwrap().clone();
        let resume = observation::resume_after_input_store(&bytecode, input);
        let source = runmat_execution::cancellation::CancellationSource::new();
        let placement = Rc::new(runmat_accelerate::placement::PlacementSession::default());
        let runtime = runmat_runtime::context::RuntimeContext::with_cancellation(
            Rc::new(runmat_runtime::execution::RuntimeExecutionService::new()),
            source.token(),
        )
        .with_service_ports(
            runmat_runtime::context::RuntimeServicePorts::default()
                .with_placement(placement.clone()),
        );
        if let Some(handle) = &borrowed {
            publish_borrowed(&runtime, &[handle]);
        }
        let probe = Rc::new(Probe::default());
        assert!(PENDING
            .with(|slot| slot.replace(Some(probe.clone())))
            .is_none());
        let _scope = Scope;
        let mut execution = Box::pin(runmat_vm::interpret_resume_in_context(
            &bytecode,
            resume,
            None,
            runtime.clone(),
        ));
        let wakes = Arc::new(Wakes::default());
        let waker = Waker::from(wakes.clone());
        let mut cx = Context::from_waker(&waker);
        assert!(execution.as_mut().poll(&mut cx).is_pending());
        assert_eq!(
            probe.polls.get(),
            1,
            "actual post-submission Single restoration hook entered"
        );
        assert_eq!(probe.drops.get(), 0);
        assert!(
            provider.buffers.lock().unwrap().len() > baseline.len(),
            "fusion owns submitted output buffers before restoration suspends"
        );
        assert_published_buffers(&runtime, provider);
        if cancel {
            source.cancel(runmat_execution::CancellationReason::User);
            assert!(
                wakes.0.load(Ordering::SeqCst) > 0,
                "provider never wakes itself"
            );
            let Poll::Ready(Err(error)) = execution.as_mut().poll(&mut cx) else {
                panic!("pending fusion must terminate on cancellation");
            };
            assert_eq!(error.identifier(), Some("RunMat:ExecutionCancelled"));
            assert_eq!(
                probe.drops.get(),
                1,
                "retained completed future released work"
            );
        }
        drop(execution);
        assert_eq!(probe.drops.get(), 1);
        let profile = placement.profile_snapshot();
        let feedback = profile
            .feedback
            .iter()
            .find(|series| series.candidate.contains("fusion.elementwise"))
            .expect("selected typed fusion reports terminal failure");
        assert_eq!(feedback.samples, 0);
        assert_eq!(feedback.failures, 1);
        assert_eq!(
            probe.polls.get(),
            1,
            "no resumed provider work or fallback replay"
        );
        assert_eq!(
            *provider.buffers.lock().unwrap(),
            baseline,
            "only invocation-owned uploads must be released"
        );
        assert_published_buffers(&runtime, provider);
        if let Some(handle) = borrowed {
            unpublish_borrowed(&runtime);
            provider.free(&handle).unwrap();
        }
        assert!(runmat_runtime::context::legacy::active().is_none());
    });
}
