//! Native search control continuation and provider ownership regression tests.
//! Actual compiled builtin call + retained provider Pending; not a synthetic Await.
use super::*;
use futures::task::AtomicWaker;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_builtins::BuiltinAsyncBehavior;
use runmat_native_executor::execute::{NativeInvocation, NativeInvocationStep};
use runmat_runtime::source_context::ProgramSourceRecord;
use std::error::Error;
use std::fmt;
use std::sync::OnceLock;
use std::task::Poll;

#[repr(usize)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum DownloadMode {
    Pending,
    Fail,
    Ready,
}

impl DownloadMode {
    fn from_atomic(value: usize) -> Self {
        match value {
            0 => Self::Pending,
            1 => Self::Fail,
            2 => Self::Ready,
            _ => panic!("fixture download mode is invalid"),
        }
    }
}

#[derive(Debug)]
struct NativeDownloadFault;

impl fmt::Display for NativeDownloadFault {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str("typed native fixture control download failure")
    }
}

impl Error for NativeDownloadFault {}

struct PendingControlProvider {
    inner: InProcessProvider,
    mode: AtomicUsize,
    downloads: AtomicUsize,
    wake: AtomicWaker,
}

impl PendingControlProvider {
    fn set_mode(&self, mode: DownloadMode) {
        self.mode.store(mode as usize, Ordering::SeqCst);
        self.wake.wake();
    }
}

impl AccelProvider for PendingControlProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, input: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(input)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            let observed_mode = futures::future::poll_fn(|cx| {
                self.wake.register(cx.waker());
                match DownloadMode::from_atomic(self.mode.load(Ordering::SeqCst)) {
                    DownloadMode::Pending => Poll::Pending,
                    mode => Poll::Ready(mode),
                }
            })
            .await;
            if observed_mode == DownloadMode::Fail {
                return Err(anyhow::Error::new(NativeDownloadFault));
            }
            self.inner.download(input).await
        })
    }
}

#[derive(Clone, Copy)]
enum SearchControlCase {
    EndsWith,
    StartsWith,
    Strfind,
}

impl SearchControlCase {
    fn name(self) -> &'static str {
        match self {
            Self::EndsWith => "endsWith",
            Self::StartsWith => "startsWith",
            Self::Strfind => "strfind",
        }
    }

    fn option(self) -> &'static str {
        match self {
            Self::EndsWith | Self::StartsWith => "IgnoreCase",
            Self::Strfind => "ForceCellOutput",
        }
    }

    fn pattern(self) -> &'static str {
        match self {
            Self::StartsWith => "run",
            Self::EndsWith | Self::Strfind => "Mat",
        }
    }

    fn assert_output(self, outputs: &[Value]) {
        assert_eq!(outputs.len(), 1);
        match self {
            Self::Strfind => {
                let Value::Cell(cell) = &outputs[0] else {
                    panic!("compiled strfind must force cell output");
                };
                assert_eq!((cell.rows, cell.cols), (1, 1));
                let Value::Tensor(indices) = cell.get(0, 0).unwrap() else {
                    panic!("strfind cell must contain Double indices");
                };
                assert_eq!(indices.shape(), &[1, 1]);
                assert_eq!(indices.materialize_f64(), vec![4.0]);
            }
            Self::EndsWith | Self::StartsWith => assert_eq!(outputs, &[Value::Bool(true)]),
        }
    }
}

fn quartet_builtin_fixture(case: SearchControlCase) -> runmat_native_codegen::NativeAssembly {
    let entry = runmat_builtins::builtin_catalog_entry_by_name(case.name())
        .expect("selected canonical search catalog entry");
    assert_eq!(entry.identity.name, case.name());
    let semantics = entry.legacy_semantics();
    assert_eq!(semantics.async_behavior, BuiltinAsyncBehavior::MaySuspend);
    let span = Span { start: 0, end: 8 };
    lower_body_with_inputs(
        "search_native_pending",
        5,
        vec![MirStmt {
            kind: MirStmtKind::Assign {
                place: MirPlace::Local(MirLocalId(4)),
                value: MirRvalue::Call(MirCall {
                    callee: MirCallee::Static(CallableIdentity::Builtin(BuiltinId(
                        entry.identity.name.into(),
                    ))),
                    args: (0..4)
                        .map(|id| MirCallArg::Single(MirOperand::Local(MirLocalId(id))))
                        .collect(),
                    arg_spans: vec![span; 4],
                    syntax: runmat_hir::CallSyntax::Plain,
                    requested_outputs: RequestedOutputCount::One,
                    fallback_policy: CallableFallbackPolicy::None,
                    workspace_first_name: None,
                    bare_identifier: false,
                    async_behavior: match semantics.async_behavior {
                        BuiltinAsyncBehavior::NeverSuspends => AsyncBehaviorFact::NeverSuspends,
                        BuiltinAsyncBehavior::MaySuspend => AsyncBehaviorFact::MaySuspend,
                        BuiltinAsyncBehavior::RequiresAsyncRuntime => {
                            AsyncBehaviorFact::RequiresAsyncRuntime
                        }
                    },
                    effects: semantics.effects,
                    workspace_effect: semantics.workspace_effect,
                    environment_effect: semantics.environment_effect,
                    purity: semantics.purity,
                    semantic_kind: semantics.semantic_kind,
                }),
            },
            span,
        }],
        MirOperand::Local(MirLocalId(4)),
        span,
        (0..4).map(BindingId).collect(),
    )
}

fn begin_case(
    executor: &NativeExecutor,
    case: SearchControlCase,
    input: &GpuTensorHandle,
) -> NativeInvocation {
    let source_text = format!(
        "function out = search_native_pending(text, pattern, option, control)\nout = {}(text, pattern, option, control);\nend",
        case.name()
    );
    let runtime = unbound_runtime_context()
        .with_program_sources([ProgramSourceRecord {
            ordinal: ProgramSourceId(0),
            name: Arc::from("search_native_pending.m"),
            fullpath_name: None,
            text: Arc::from(source_text),
        }])
        .expect("issue exact native fixture source lease");
    assert!(runtime
        .admitted_program_source_id(ProgramSourceId(0))
        .expect("selected source ordinal is valid")
        .is_some());
    // The native executor scopes the call to this request context. Ambient
    // compatibility policy is not the authority for a retained invocation.
    runtime.set_runmat_extensions_enabled(true);
    executor
        .begin_request(runmat_native_executor::NativeInvocationRequest {
            function: ProgramFunctionId(0),
            captures: Vec::new(),
            arguments: vec![
                Value::String("RunMat".into()),
                Value::String(case.pattern().into()),
                Value::String(case.option().into()),
                Value::GpuTensor(input.clone()),
            ],
            requested_outputs: 1,
            runtime,
            deoptimization: DeoptimizationPolicy::default(),
            osr_target: None,
            workspace: None,
        })
        .unwrap()
}

fn suspension(invocation: &mut NativeInvocation) -> (u64, u64) {
    let NativeInvocationStep::Suspended {
        continuation,
        generation,
    } = invocation.advance().unwrap()
    else {
        panic!("compiled resident-control call must suspend at its download");
    };
    (continuation, generation)
}

#[test]
fn compiled_search_resident_controls_suspend_resume_fail_and_drop_without_replay() {
    static PROVIDER: OnceLock<PendingControlProvider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| PendingControlProvider {
        inner: InProcessProvider::new(),
        mode: AtomicUsize::new(DownloadMode::Pending as usize),
        downloads: AtomicUsize::new(0),
        wake: AtomicWaker::new(),
    });
    let _provider = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let weak = input.allocation_lease().downgrade();
    let descriptor = input.wire_descriptor();
    for case in [
        SearchControlCase::EndsWith,
        SearchControlCase::StartsWith,
        SearchControlCase::Strfind,
    ] {
        let executor = compile_executor(quartet_builtin_fixture(case)).unwrap();

        provider.set_mode(DownloadMode::Pending);
        let before = provider.downloads.load(Ordering::SeqCst);
        let mut success = begin_case(&executor, case, &input);
        let (continuation, generation) = suspension(&mut success);
        // Native entry installs an unpolled owned continuation. Poll the
        // exact resume future once to observe the provider's Pending download.
        let mut resume = Box::pin(success.resume_suspension(continuation, generation));
        assert!(
            futures::executor::block_on(async { futures::poll!(resume.as_mut()) }).is_pending()
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 1);
        assert!(weak.acquire().is_some());
        provider.set_mode(DownloadMode::Ready);
        futures::executor::block_on(resume).unwrap();
        let NativeInvocationStep::Completed(execution) = success.advance().unwrap() else {
            panic!("native continuation must complete the exact selected search call");
        };
        case.assert_output(&execution.outputs);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 1);
        drop(success);

        provider.set_mode(DownloadMode::Pending);
        let mut cancelled = begin_case(&executor, case, &input);
        let (continuation, generation) = suspension(&mut cancelled);
        let mut resume = Box::pin(cancelled.resume_suspension(continuation, generation));
        assert!(
            futures::executor::block_on(async { futures::poll!(resume.as_mut()) }).is_pending()
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 2);
        drop(resume);
        drop(cancelled);
        provider.set_mode(DownloadMode::Ready);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 2);
        assert!(weak.acquire().is_some());

        provider.set_mode(DownloadMode::Pending);
        let mut failed = begin_case(&executor, case, &input);
        let (continuation, generation) = suspension(&mut failed);
        let mut resume = Box::pin(failed.resume_suspension(continuation, generation));
        assert!(
            futures::executor::block_on(async { futures::poll!(resume.as_mut()) }).is_pending()
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 3);
        provider.set_mode(DownloadMode::Fail);
        let error = futures::executor::block_on(resume).unwrap_err();
        let NativeExecutorError::Runtime(runtime) = error else {
            panic!("native resumed failure must preserve typed RuntimeError");
        };
        assert_eq!(runtime.identifier(), Some("RunMat:gather:DownloadFailed"));
        let mut source = Error::source(runtime.as_ref());
        let mut original_found = false;
        while let Some(cause) = source {
            original_found |= cause.is::<NativeDownloadFault>();
            source = cause.source();
        }
        assert!(
            original_found,
            "original native provider cause must survive"
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 3);
        drop(failed);
        assert!(weak.acquire().is_some());
        assert_eq!(input.wire_descriptor(), descriptor);
    }
    drop(input);
    assert!(weak.acquire().is_none());
}
