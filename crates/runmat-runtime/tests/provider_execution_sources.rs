//! Actual domain dispatch keeps terminal, typed provider causes.
#![cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;
use runmat_accelerate_api::test_support::TestAllocationOwner;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct DeviceFailure;
impl std::fmt::Display for DeviceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("injected transport failure")
    }
}
impl std::error::Error for DeviceFailure {}

struct Provider {
    owner: TestAllocationOwner,
    calls: AtomicUsize,
    nested: bool,
}
impl Provider {
    fn fail(&self) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let cause = if self.nested {
            anyhow::Error::new(
                runmat_runtime::build_runtime_error("authoritative provider contract")
                    .with_identifier("RunMat:test:ProviderContract")
                    .with_builtin("provider-origin")
                    .with_source(DeviceFailure)
                    .build(),
            )
        } else {
            anyhow::Error::new(DeviceFailure)
        };
        Err(cause)
    }
}
macro_rules! dimension_hook {
    ($name:ident) => {
        fn $name<'a>(
            &'a self,
            _: &'a GpuTensorHandle,
            _: usize,
        ) -> AccelProviderFuture<'a, GpuTensorHandle> {
            Box::pin(async move { self.fail() })
        }
    };
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "terminal provider source fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("failure may not retry upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("failure may not gather")
    }
    dimension_hook!(reduce_sum_dim);
    dimension_hook!(reduce_mean_dim);
    dimension_hook!(reduce_integer_mean_native_dim);
    fn unary_cos<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail() })
    }
    fn moving_window<'a>(
        &'a self,
        _: &'a ProviderMovingWindowRequest<'a>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.fail() })
    }
    fn gradient_dim(
        &self,
        _: &GpuTensorHandle,
        _: usize,
        _: f64,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.fail()
    }
    fn diff_dim(&self, _: &GpuTensorHandle, _: usize, _: usize) -> anyhow::Result<GpuTensorHandle> {
        self.fail()
    }
    fn cumtrapz_dim(
        &self,
        _: &GpuTensorHandle,
        _: usize,
        _: ProviderTrapezoidSpacing<'_>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.fail()
    }
}

#[test]
fn actual_provider_domains_retain_typed_causes_without_retry() {
    runmat_gc::gc_test_context(|| {
        for nested in [false, true] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                calls: AtomicUsize::new(0),
                nested,
            }));
            let _provider = ThreadProviderGuard::set(Some(provider));
            for (name, integer) in [
                ("movsum", false),
                ("gradient", false),
                ("cumtrapz", false),
                ("diff", false),
                ("sum", false),
                ("mean", false),
                ("mean", true),
                ("pol2cart", false),
            ] {
                let input = provider
                    .owner
                    .allocate_zeroed(
                        vec![2, 2],
                        if integer {
                            NumericElementType::I64
                        } else {
                            NumericElementType::F64
                        },
                        GpuTensorStorage::Real,
                    )
                    .unwrap();
                let handle = input
                    .handle
                    .with_provenance(GpuHandleProvenance::Automatic)
                    .unwrap();
                let mut args = vec![Value::GpuTensor(handle)];
                match name {
                    "pol2cart" => args.push(args[0].clone()),
                    "movsum" => args.push(Value::Num(2.0)),
                    "mean" | "sum" if integer => {
                        args.push(Value::Num(1.0));
                        args.push(Value::from("native"));
                    }
                    "mean" | "sum" => args.push(Value::Num(1.0)),
                    _ => {}
                }
                let before = provider.calls.load(Ordering::SeqCst);
                let error = block_on(call_builtin_async_with_outputs(name, &args, 1)).unwrap_err();
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    before + 1,
                    "{name}: actual selected hook"
                );
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never, "{name}");
                let mut cursor: &(dyn std::error::Error + 'static) = &error;
                loop {
                    if cursor.is::<DeviceFailure>() {
                        break;
                    }
                    cursor = cursor
                        .source()
                        .unwrap_or_else(|| panic!("{name}: typed cause missing: {error:?}"));
                }
                if nested {
                    assert_eq!(error.identifier(), Some("RunMat:test:ProviderContract"));
                    assert_eq!(error.message(), "authoritative provider contract");
                    assert_eq!(error.context.builtin.as_deref(), Some("provider-origin"));
                }
                assert_eq!(input.probe.retirements(), 0);
                drop(error);
                drop(args);
                drop(input.handle);
                assert_eq!(input.probe.retirements(), 1);
            }
        }
    });
}
