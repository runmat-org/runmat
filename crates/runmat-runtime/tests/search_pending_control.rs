//! Candidate `crates/runmat-runtime/tests/search_pending_control.rs`.
//! Actual retained provider and dispatcher, not a ready mock or direct parser.
#![cfg(not(target_arch = "wasm32"))]
use futures::{
    executor::block_on,
    task::{waker, ArcWake, AtomicWaker},
    FutureExt,
};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use runmat_value::Value;
use std::error::Error;
use std::fmt;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll};

#[path = "support/retained_provider.rs"]
mod retained_provider;

#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
struct DownloadFault;

impl fmt::Display for DownloadFault {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str("typed injected control download failure")
    }
}

impl Error for DownloadFault {}

struct WakeCounter(AtomicUsize);

impl ArcWake for WakeCounter {
    fn wake_by_ref(this: &Arc<Self>) {
        this.0.fetch_add(1, Ordering::SeqCst);
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

    fn assert_output(self, result: &Value) {
        match self {
            Self::Strfind => {
                let Value::Cell(cell) = result else {
                    panic!("strfind must force cell output");
                };
                assert_eq!((cell.rows, cell.cols), (1, 1));
                let Value::Tensor(indices) = cell.get(0, 0).unwrap() else {
                    panic!("strfind cell must contain Double indices");
                };
                assert_eq!(indices.shape(), &[1, 1]);
                assert_eq!(indices.materialize_f64(), vec![4.0]);
            }
            Self::EndsWith | Self::StartsWith => assert_eq!(result, &Value::Bool(true)),
        }
    }
}

struct SuspendedControlProvider {
    inner: InProcessProvider,
    mode: AtomicUsize,
    downloads: AtomicUsize,
    wake: AtomicWaker,
}

impl SuspendedControlProvider {
    fn set_mode(&self, mode: DownloadMode) {
        self.mode.store(mode as usize, Ordering::SeqCst);
        self.wake.wake();
    }
}

impl AccelProvider for SuspendedControlProvider {
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
                return Err(anyhow::Error::new(DownloadFault));
            }
            self.inner.download(input).await
        })
    }
}

#[test]
fn search_resident_control_pending_drop_error_and_success_keep_borrowed_input() {
    let provider = retained_provider::retained_test_provider!(PROVIDER: SuspendedControlProvider = SuspendedControlProvider {
        inner: InProcessProvider::new(),
        mode: AtomicUsize::new(DownloadMode::Pending as usize),
        downloads: AtomicUsize::new(0),
        wake: AtomicWaker::new(),
    });
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
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
        let name = case.name();
        provider.set_mode(DownloadMode::Pending);
        let before = provider.downloads.load(Ordering::SeqCst);
        let wake_counter = Arc::new(WakeCounter(AtomicUsize::new(0)));
        let real_waker = waker(wake_counter.clone());
        let invalid = vec![
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            Value::String("NotAnOption".into()),
            Value::GpuTensor(input.clone()),
        ];
        let mut rejected = Box::pin(runmat_runtime::call_builtin_async(name, &invalid));
        assert!(matches!(
            rejected.poll_unpin(&mut Context::from_waker(&real_waker)),
            Poll::Ready(Err(_))
        ));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before);
        drop(rejected);

        let args = vec![
            Value::String("RunMat".into()),
            Value::String(case.pattern().into()),
            Value::String(case.option().into()),
            Value::GpuTensor(input.clone()),
        ];
        let mut future = Box::pin(runmat_runtime::call_builtin_async(name, &args));
        assert!(matches!(
            future.poll_unpin(&mut Context::from_waker(&real_waker)),
            Poll::Pending
        ));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 1);
        drop(future);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 1);
        assert!(weak.acquire().is_some());
        assert_eq!(input.wire_descriptor(), descriptor);

        provider.set_mode(DownloadMode::Fail);
        let failure = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
        assert_eq!(failure.identifier(), Some("RunMat:gather:DownloadFailed"));
        let mut source = Error::source(&failure);
        let mut original_found = false;
        while let Some(cause) = source {
            original_found |= cause.is::<DownloadFault>();
            source = cause.source();
        }
        assert!(original_found, "{name}: typed provider cause must survive");
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 2);
        assert!(weak.acquire().is_some());
        // Resume the *same* pending future after a real waker notification.
        provider.set_mode(DownloadMode::Pending);
        let mut resumed = Box::pin(runmat_runtime::call_builtin_async(name, &args));
        assert!(matches!(
            resumed.poll_unpin(&mut Context::from_waker(&real_waker)),
            Poll::Pending
        ));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 3);
        let wakes_before = wake_counter.0.load(Ordering::SeqCst);
        provider.set_mode(DownloadMode::Ready);
        assert!(wake_counter.0.load(Ordering::SeqCst) > wakes_before);
        let Poll::Ready(Ok(result)) = resumed.poll_unpin(&mut Context::from_waker(&real_waker))
        else {
            panic!("{name}: same future must complete after wake");
        };
        case.assert_output(result.value());
        drop(resumed);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), before + 3);
        assert!(weak.acquire().is_some());
        assert_eq!(input.wire_descriptor(), descriptor);
        drop(args);
        drop(invalid);
    }
    drop(input);
    assert!(weak.acquire().is_none());
}
