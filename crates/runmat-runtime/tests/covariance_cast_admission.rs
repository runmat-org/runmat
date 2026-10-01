use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug)]
enum Mode {
    Valid,
    RawShape,
    RawInteger,
    ConvertedShape,
}

struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    casts: AtomicUsize,
    downloads: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
}

impl Provider {
    fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            mode,
            casts: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            outputs: Mutex::new(Vec::new()),
        }))
    }

    fn result(&self, converted: bool) -> anyhow::Result<GpuTensorHandle> {
        let shape = if matches!(self.mode, Mode::RawShape) && !converted
            || matches!(self.mode, Mode::ConvertedShape) && converted
        {
            vec![1, 4]
        } else {
            vec![2, 2]
        };
        let element = if matches!(self.mode, Mode::RawInteger) && !converted {
            NumericElementType::I32
        } else if converted {
            NumericElementType::F32
        } else {
            NumericElementType::F64
        };
        let allocation = self
            .owner
            .allocate_zeroed(shape, element, GpuTensorStorage::Real)?;
        self.outputs.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "covariance cast admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }

    fn covariance<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: Option<&'a GpuTensorHandle>,
        _: Option<&'a GpuTensorHandle>,
        _: &'a CovarianceOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.result(false)
                .map_err(ProviderExecutionError::from_source)
        })
    }

    fn corrcoef<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a CorrcoefOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.result(false)
                .map_err(ProviderExecutionError::from_source)
        })
    }

    fn unary_single<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.casts.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { self.result(true) })
    }
}

#[test]
fn covariance_family_validates_raw_results_before_conversion_and_converted_results_after() {
    for builtin in ["cov", "corrcoef"] {
        for mode in [
            Mode::RawShape,
            Mode::RawInteger,
            Mode::ConvertedShape,
            Mode::Valid,
        ] {
            let provider = Provider::new(mode);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 2], NumericElementType::F32, GpuTensorStorage::Real)
                .unwrap();
            let result = block_on(call_builtin_async_with_outputs(
                builtin,
                &[Value::GpuTensor(input.handle.clone())],
                1,
            ));
            if matches!(mode, Mode::Valid) {
                let output = match result.unwrap() {
                    Value::GpuTensor(output) => output,
                    Value::OutputList(values) if values.len() == 1 => {
                        let Value::GpuTensor(output) = values.into_iter().next().unwrap() else {
                            panic!("{builtin} should preserve resident output");
                        };
                        output
                    }
                    _ => panic!("{builtin} should preserve resident output"),
                };
                assert_eq!(handle_precision(&output), Some(ProviderPrecision::F32));
                assert_eq!(output.shape(), &[2, 2]);
                assert!(!output.shares_allocation_with(&input.handle));
                drop(output);
            } else {
                let error = result.unwrap_err();
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                assert_eq!(
                    error.identifier(),
                    Some(if builtin == "cov" {
                        "RunMat:cov:Internal"
                    } else {
                        "RunMat:corrcoef:Internal"
                    })
                );
            }
            assert_eq!(
                provider.casts.load(Ordering::SeqCst),
                if matches!(mode, Mode::RawShape | Mode::RawInteger) {
                    0
                } else {
                    1
                },
                "{builtin} {mode:?}"
            );
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            assert!(input.handle.allocation_lease().validate_live().is_ok());
            assert_eq!(input.probe.retirements(), 0);
            let probes = provider.outputs.lock().unwrap();
            assert!(probes.iter().all(|probe| probe.retirements() == 1));
        }
    }
}
