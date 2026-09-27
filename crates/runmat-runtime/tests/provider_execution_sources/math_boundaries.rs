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
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        assert!(
            matches!(self.failure, "upload_x" | "upload_query" | "interp1"),
            "terminal hook failure must not upload"
        );
        let hook = if self.trace.lock().unwrap().is_empty() {
            "upload_x"
        } else {
            "upload_query"
        };
        if hook == self.failure {
            return self.fail(hook);
        }
        self.trace.lock().unwrap().push(hook);
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(view.data.to_vec()),
            shape: view.shape.to_vec(),
            storage: GpuTensorStorage::Real,
        })?;
        self.probes
            .lock()
            .unwrap()
            .push((allocation.handle.buffer_id(), allocation.probe));
        Ok(allocation.handle)
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
            if self.failure == "reduce_min" {
                return self.fail("reduce_min");
            }
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
    fn interp1<'a>(
        &'a self,
        _: &'a ProviderInterp1Request<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("interp1") })
    }
    fn unary_sin<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.trig("unary_sin", input) })
    }
    fn unary_cos<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.trig("unary_cos", input) })
    }
    fn unary_tan<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.trig("unary_tan", input) })
    }
    fn reduce_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("reduce_max") })
    }
    fn transpose(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        if self.failure == "transpose" {
            return self.fail("transpose");
        }
        assert_eq!(self.failure, "unary_conj");
        self.trace.lock().unwrap().push("transpose");
        let mut shape = input.shape().to_vec();
        shape.swap(0, 1);
        self.allocate(shape, handle_storage(input))
    }
    fn permute(&self, _: &GpuTensorHandle, _: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        self.fail("permute")
    }
    fn scalar_mul(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.fail("scalar_mul")
    }
    fn matmul<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("matmul") })
    }
    fn dot<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("dot") })
    }
    fn cross(
        &self,
        _: &GpuTensorHandle,
        _: &GpuTensorHandle,
        _: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.fail("cross")
    }
    fn complex_from_real_imag<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail("complex_from_real_imag") })
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

#[test]
fn matrix_dispatch_preserves_each_provider_boundary_and_owned_transpose() {
    for (name, hook, shape) in [
        ("dot", "dot", vec![3, 1]),
        ("cross", "cross", vec![3, 1]),
        ("mtimes", "matmul", vec![2, 2]),
    ] {
        check_case(
            name,
            hook,
            &[hook],
            shape,
            GpuTensorStorage::Real,
            |left, right| {
                vec![
                    Value::GpuTensor(left.clone()),
                    Value::GpuTensor(right.clone()),
                ]
            },
        );
    }
    for scalar_first in [false, true] {
        check_case(
            "mtimes",
            "scalar_mul",
            &["scalar_mul"],
            vec![2, 2],
            GpuTensorStorage::Real,
            |left, _| {
                let resident = Value::GpuTensor(left.clone());
                if scalar_first {
                    vec![Value::Num(2.0), resident]
                } else {
                    vec![resident, Value::Num(2.0)]
                }
            },
        );
    }
    for (hook, shape, storage, trace) in [
        (
            "transpose",
            vec![2, 3],
            GpuTensorStorage::Real,
            vec!["transpose"],
        ),
        (
            "permute",
            vec![2, 3, 2],
            GpuTensorStorage::Real,
            vec!["permute"],
        ),
        (
            "unary_conj",
            vec![2, 3],
            GpuTensorStorage::ComplexInterleaved,
            vec!["transpose", "unary_conj"],
        ),
    ] {
        check_case("ctranspose", hook, &trace, shape, storage, |left, _| {
            vec![Value::GpuTensor(left.clone())]
        });
    }
}
