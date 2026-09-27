//! One authentic owner and trace fixture for terminal math provider boundaries.
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    *,
};
use runmat_value::Value;
use std::sync::Mutex;

#[derive(Debug)]
struct BoundaryFailure(&'static str);
impl std::fmt::Display for BoundaryFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "injected boundary failure: {}", self.0)
    }
}
impl std::error::Error for BoundaryFailure {}

struct Provider {
    owner: TestAllocationOwner,
    trace: Mutex<Vec<&'static str>>,
    probes: Mutex<Vec<(u64, RetirementProbe)>>,
}
impl Provider {
    fn fail<T>(&self, hook: &'static str) -> anyhow::Result<T> {
        self.trace.lock().unwrap().push(hook);
        Err(anyhow::Error::new(BoundaryFailure(hook)))
    }
    fn read_probe(&self, handle: &GpuTensorHandle) -> anyhow::Result<Vec<f64>> {
        self.trace.lock().unwrap().push("download_probe");
        let probe = self
            .probes
            .lock()
            .unwrap()
            .iter()
            .find(|(id, _)| *id == handle.buffer_id())
            .expect("only the admitted domain probe may be downloaded")
            .1
            .clone();
        probe.with_data(handle, |data| match data {
            HostNumericDataView::F64(data) => data.to_vec(),
            _ => panic!("probe must retain actual F64 storage"),
        })
    }
}
macro_rules! unary {
    ($hook:ident) => {
        fn $hook<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move { self.fail(stringify!($hook)) })
        }
    };
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "math boundary source witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("terminal hook failure must not upload")
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostTensorOwned {
                data: self.read_probe(handle)?,
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(self.read_probe(handle)?),
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn reduce_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.trace.lock().unwrap().push("reduce_min");
            let result = self.owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })?;
            self.probes
                .lock()
                .unwrap()
                .push((result.handle.buffer_id(), result.probe));
            Ok(result.handle)
        })
    }
    unary!(unary_erf);
    unary!(unary_erfcinv);
    unary!(unary_factorial);
    unary!(unary_gamma);
    unary!(unary_gammaln);
    unary!(unary_sqrt);
    unary!(unary_pow2);
    unary!(unary_nextpow2);
    unary!(unary_log);
    unary!(unary_log2);
    unary!(unary_log10);
    unary!(unary_log1p);
    unary!(unary_real);
    unary!(unary_imag);
    unary!(unary_conj);
    unary!(complex_from_real);
    fn complex_from_real_imag<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("complex_from_real_imag") })
    }
}

fn check(name: &str, hook: &'static str, probe: bool, binary: bool) {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        trace: Mutex::new(Vec::new()),
        probes: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let left = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let right = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut args = vec![Value::GpuTensor(left.handle.clone())];
    if binary {
        args.push(Value::GpuTensor(right.handle.clone()));
    }
    let error =
        futures::executor::block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
    assert_eq!(
        error.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never,
        "{name}"
    );
    let mut source: &(dyn std::error::Error + 'static) = &error;
    let mut carrier = false;
    loop {
        carrier |= source.is::<ProviderExecutionError>();
        if let Some(cause) = source.downcast_ref::<BoundaryFailure>() {
            assert_eq!(cause.0, hook, "{name}");
            break;
        }
        source = source
            .source()
            .unwrap_or_else(|| panic!("{name}: original typed cause lost: {error:?}"));
    }
    assert!(carrier, "{name}: provider execution carrier missing");
    let expected = if probe {
        vec!["reduce_min", "download_probe", hook]
    } else {
        vec![hook]
    };
    assert_eq!(*provider.trace.lock().unwrap(), expected, "{name}");
    assert!(provider
        .probes
        .lock()
        .unwrap()
        .iter()
        .all(|(_, p)| p.retirements() == 1));
    for handle in [&left.handle, &right.handle] {
        handle.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(left.probe.retirements(), 0);
    assert_eq!(right.probe.retirements(), 0);
    drop(error);
    drop(args);
    drop(left.handle);
    drop(right.handle);
    assert_eq!(left.probe.retirements(), 1);
    assert_eq!(right.probe.retirements(), 1);
}

#[test]
fn unary_and_complex_dispatch_preserve_exact_terminal_causes() {
    for (name, hook, probe) in [
        ("erf", "unary_erf", false),
        ("erfcinv", "unary_erfcinv", false),
        ("factorial", "unary_factorial", false),
        ("gamma", "unary_gamma", false),
        ("gammaln", "unary_gammaln", true),
        ("sqrt", "unary_sqrt", true),
        ("realsqrt", "unary_sqrt", true),
        ("pow2", "unary_pow2", false),
        ("nextpow2", "unary_nextpow2", false),
        ("log", "unary_log", true),
        ("log2", "unary_log2", true),
        ("log10", "unary_log10", true),
        ("log1p", "unary_log1p", true),
        ("real", "unary_real", false),
        ("imag", "unary_imag", false),
        ("conj", "unary_conj", false),
        ("complex", "complex_from_real", false),
    ] {
        check(name, hook, probe, false);
    }
    check("complex", "complex_from_real_imag", false, true);
}
