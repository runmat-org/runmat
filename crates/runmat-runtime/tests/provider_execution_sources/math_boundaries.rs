//! One authentic owner and trace fixture for terminal math provider boundaries.
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    *,
};
use runmat_value::Value;
use std::sync::Mutex;

#[path = "math_boundaries/trigonometry.rs"]
mod trigonometry;

#[path = "math_boundaries/interpolation.rs"]
mod interpolation;

#[path = "math_boundaries/hooks.rs"]
mod hooks;
#[path = "math_boundaries/operations.rs"]
mod operations;

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
    failure: &'static str,
    trace: Mutex<Vec<&'static str>>,
    probes: Mutex<Vec<(u64, RetirementProbe)>>,
}
impl Provider {
    fn trig(&self, hook: &'static str, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        if self.failure == "complex_from_real" {
            self.trace.lock().unwrap().push(hook);
            self.allocate(input.shape().to_vec(), GpuTensorStorage::Real)
        } else {
            self.fail(hook)
        }
    }
    fn allocate(
        &self,
        shape: Vec<usize>,
        storage: GpuTensorStorage,
    ) -> anyhow::Result<GpuTensorHandle> {
        let result = self
            .owner
            .allocate_zeroed(shape, NumericElementType::F64, storage)?;
        self.probes
            .lock()
            .unwrap()
            .push((result.handle.buffer_id(), result.probe));
        Ok(result.handle)
    }
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

fn check(name: &str, hook: &'static str, probe: bool, binary: bool) {
    let expected = if probe {
        vec!["reduce_min", "download_probe", hook]
    } else {
        vec![hook]
    };
    check_case(
        name,
        hook,
        &expected,
        vec![2, 1],
        GpuTensorStorage::Real,
        |left, right| {
            let mut args = vec![Value::GpuTensor(left.clone())];
            if binary {
                args.push(Value::GpuTensor(right.clone()));
            }
            args
        },
    );
}

fn check_case(
    name: &str,
    hook: &'static str,
    expected: &[&str],
    shape: Vec<usize>,
    storage: GpuTensorStorage,
    arguments: impl FnOnce(&GpuTensorHandle, &GpuTensorHandle) -> Vec<Value>,
) {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        failure: hook,
        trace: Mutex::new(Vec::new()),
        probes: Mutex::new(Vec::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let left = provider
        .owner
        .allocate_zeroed(shape.clone(), NumericElementType::F64, storage)
        .unwrap();
    let right = provider
        .owner
        .allocate_zeroed(
            shape,
            NumericElementType::F64,
            if hook == "complex_from_real" && matches!(name, "sin" | "cos" | "tan") {
                GpuTensorStorage::ComplexInterleaved
            } else {
                storage
            },
        )
        .unwrap();
    let args = arguments(&left.handle, &right.handle);
    let error =
        futures::executor::block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
    assert_eq!(
        error.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never,
        "{name} targeting {hook}: {error:?}; trace={:?}",
        provider.trace.lock().unwrap()
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
