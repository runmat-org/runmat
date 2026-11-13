use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async_with_outputs, GpuGatherRetry};
use runmat_value::{NumericDType, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[path = "covariance_cast_admission/failures.rs"]
mod failures;

#[derive(Clone, Copy, Debug)]
enum Mode {
    Valid,
    RawShape,
    RawInteger,
    RawLogical,
    RawOwner,
    RawBorrowAlias,
    ConvertedShape,
    ConvertedClass,
    ConvertedBorrowAlias,
    CastFault,
    CastUnsupported,
    WidenUnsupported,
    ProducedDownloadFault,
    RestoreUnsupported,
    RestoreFault,
    PendingProducedDownload,
}

impl Mode {
    fn materializes_produced_result(self) -> bool {
        matches!(
            self,
            Self::CastUnsupported
                | Self::WidenUnsupported
                | Self::ProducedDownloadFault
                | Self::RestoreUnsupported
                | Self::RestoreFault
                | Self::PendingProducedDownload
        )
    }
}

struct Provider {
    owner: TestAllocationOwner,
    foreign: TestAllocationOwner,
    mode: Mode,
    casts: AtomicUsize,
    covariance_calls: AtomicUsize,
    corrcoef_calls: AtomicUsize,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
    input_downloads: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
    borrowed: Mutex<Option<allocation::WeakGpuTensorHandle>>,
    input_probe: Mutex<Option<RetirementProbe>>,
}

impl Provider {
    fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            foreign: TestAllocationOwner::new(),
            mode,
            casts: AtomicUsize::new(0),
            covariance_calls: AtomicUsize::new(0),
            corrcoef_calls: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            input_downloads: AtomicUsize::new(0),
            outputs: Mutex::new(Vec::new()),
            borrowed: Mutex::new(None),
            input_probe: Mutex::new(None),
        }))
    }

    fn result(&self, converted: bool, raw_value: f64) -> anyhow::Result<GpuTensorHandle> {
        if matches!(self.mode, Mode::ConvertedBorrowAlias) && converted {
            return self
                .borrowed
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .acquire()
                .map_err(Into::into);
        }
        if matches!(self.mode, Mode::RawLogical) && !converted {
            let allocation = self.owner.allocate_logical(vec![0; 4], vec![2, 2])?;
            self.outputs.lock().unwrap().push(allocation.probe);
            return Ok(allocation.handle);
        }
        let shape = if matches!(self.mode, Mode::RawShape) && !converted
            || matches!(self.mode, Mode::ConvertedShape) && converted
        {
            vec![1, 4]
        } else {
            vec![2, 2]
        };
        let element = if matches!(self.mode, Mode::RawInteger) && !converted {
            NumericElementType::I32
        } else if converted && !matches!(self.mode, Mode::ConvertedClass) {
            NumericElementType::F32
        } else {
            NumericElementType::F64
        };
        let owner = if matches!(self.mode, Mode::RawOwner) && !converted {
            &self.foreign
        } else {
            &self.owner
        };
        let allocation = if matches!(self.mode, Mode::WidenUnsupported) && !converted {
            owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(vec![raw_value as f32; 4]),
                shape,
                storage: GpuTensorStorage::Real,
            })?
        } else if self.mode.materializes_produced_result() && !converted {
            owner.allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![raw_value; 4]),
                shape,
                storage: GpuTensorStorage::Real,
            })?
        } else {
            owner.allocate_zeroed(shape, element, GpuTensorStorage::Real)?
        };
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

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let data = match host.data {
            HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
            HostNumericDataView::F64(values) if matches!(self.mode, Mode::WidenUnsupported) => {
                HostNumericDataOwned::F64(values.to_vec())
            }
            _ => {
                return Err(
                    anyhow::anyhow!("restoration requires the selected output class").into(),
                )
            }
        };
        if matches!(self.mode, Mode::RestoreUnsupported) {
            return Err(unsupported_provider_operation(
                "restoring the produced covariance is unsupported",
            )
            .into());
        }
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.outputs.lock().unwrap().push(allocation.probe);
        if matches!(self.mode, Mode::RestoreFault) {
            return Err(ProviderExecutionError::from_source(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "restoring the produced covariance failed",
            ))
            .after_effect());
        }
        Ok(allocation.handle)
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("unexpected download") })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        if self
            .input_probe
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|probe| probe.with_data(handle, |_| ()).is_ok())
        {
            self.input_downloads.fetch_add(1, Ordering::SeqCst);
            return Box::pin(async { anyhow::bail!("input download would replay covariance") });
        }
        if matches!(self.mode, Mode::ProducedDownloadFault) {
            return Box::pin(async {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "produced result download failed",
                )
                .into())
            });
        }
        if matches!(self.mode, Mode::PendingProducedDownload) {
            return Box::pin(futures::future::pending());
        }
        let probes = self.outputs.lock().unwrap().clone();
        Box::pin(async move {
            let data = probes
                .iter()
                .find_map(|probe| {
                    probe
                        .with_data(handle, |view| match view {
                            HostNumericDataView::F32(values) => {
                                Some(HostNumericDataOwned::F32(values.to_vec()))
                            }
                            HostNumericDataView::F64(values)
                                if self.mode.materializes_produced_result() =>
                            {
                                Some(HostNumericDataOwned::F64(values.to_vec()))
                            }
                            _ => None,
                        })
                        .ok()
                        .flatten()
                })
                .ok_or_else(|| anyhow::anyhow!("unexpected numeric download"))?;
            Ok(HostNumericTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: handle.physical().storage(),
            })
        })
    }

    fn covariance<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: Option<&'a GpuTensorHandle>,
        _: Option<&'a GpuTensorHandle>,
        _: &'a CovarianceOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.covariance_calls.fetch_add(1, Ordering::SeqCst);
        *self.borrowed.lock().unwrap() = Some(input.downgrade());
        Box::pin(async move {
            if matches!(self.mode, Mode::RawBorrowAlias) {
                return Ok(input.clone());
            }
            self.result(false, 0.5).map_err(Into::into)
        })
    }

    fn corrcoef<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: &'a CorrcoefOptions,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.corrcoef_calls.fetch_add(1, Ordering::SeqCst);
        *self.borrowed.lock().unwrap() = Some(input.downgrade());
        Box::pin(async move {
            if matches!(self.mode, Mode::RawBorrowAlias) {
                return Ok(input.clone());
            }
            self.result(false, 1.0).map_err(Into::into)
        })
    }

    fn unary_single<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.casts.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if matches!(self.mode, Mode::CastFault) {
                return Err(anyhow::anyhow!("cast execution failed").into());
            }
            if self.mode.materializes_produced_result() {
                return Err(
                    unsupported_provider_operation("fixture cannot cast this result").into(),
                );
            }
            self.result(true, 0.0).map_err(Into::into)
        })
    }

    fn unary_double<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.casts.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if matches!(self.mode, Mode::WidenUnsupported) {
                Err(unsupported_provider_operation("fixture cannot widen this result").into())
            } else {
                self.result(true, 0.0).map_err(Into::into)
            }
        })
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_unsupported_cast_respects_each_builtin_host_boundary_and_input_lifetime() {
    for builtin in ["cov", "corrcoef"] {
        let provider = Provider::new(Mode::CastUnsupported);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0]),
                shape: vec![2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        *provider.input_probe.lock().unwrap() = Some(input.probe.clone());
        let result = block_on(call_builtin_async_with_outputs(
            builtin,
            &[Value::GpuTensor(input.handle.clone())],
            1,
        ))
        .unwrap();
        let output = match result.value() {
            Value::OutputList(values) if values.len() == 1 => &values[0],
            value => value,
        };
        assert_eq!(provider.casts.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.covariance_calls.load(Ordering::SeqCst),
            usize::from(builtin == "cov")
        );
        assert_eq!(
            provider.corrcoef_calls.load(Ordering::SeqCst),
            usize::from(builtin == "corrcoef")
        );
        // The only pre-publication download is the already-computed F64
        // result. The fixture rejects downloading the F32 source, so a whole
        // covariance/correlation replay cannot satisfy this oracle.
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(provider.input_downloads.load(Ordering::SeqCst), 0);
        match (builtin, output) {
            ("cov", Value::GpuTensor(output)) => {
                assert_eq!(output.shape(), &[2, 2]);
                assert_eq!(handle_precision(output), Some(ProviderPrecision::F32));
                assert!(!output.shares_allocation_with(&input.handle));
                assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
                let downloaded = block_on(provider.download_numeric(output)).unwrap();
                let HostNumericDataOwned::F32(values) = downloaded.data else {
                    panic!("cov restoration changed numeric precision");
                };
                assert_eq!(downloaded.shape, vec![2, 2]);
                assert!(values.iter().all(|&value| (value - 0.5).abs() < 1e-6));
            }
            ("corrcoef", Value::Tensor(output)) => {
                // This builtin may finish on host, but never recomputes the
                // completed correlation after the cast hook refuses it.
                assert_eq!(output.shape(), &[2, 2]);
                assert_eq!(output.numeric_dtype(), NumericDType::F32);
                assert!(output
                    .materialize_f64()
                    .iter()
                    .all(|&value| (value - 1.0).abs() < 1e-6));
                assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            }
            _ => panic!("{builtin} fallback returned the wrong output carrier"),
        }
        drop(result);
        assert!(input.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(input.probe.retirements(), 0);
        assert!(provider
            .outputs
            .lock()
            .unwrap()
            .iter()
            .all(|probe| probe.retirements() == 1));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn covariance_family_validates_raw_results_before_conversion_and_converted_results_after() {
    for builtin in ["cov", "corrcoef"] {
        for mode in [
            Mode::RawShape,
            Mode::RawInteger,
            Mode::RawLogical,
            Mode::RawOwner,
            Mode::RawBorrowAlias,
            Mode::ConvertedShape,
            Mode::ConvertedClass,
            Mode::ConvertedBorrowAlias,
            Mode::CastFault,
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
                let rooted = result.unwrap();
                let output = match rooted.value() {
                    Value::GpuTensor(output) => output,
                    Value::OutputList(values) if values.len() == 1 => {
                        let Value::GpuTensor(output) = &values[0] else {
                            panic!("{builtin} should preserve resident output");
                        };
                        output
                    }
                    _ => panic!("{builtin} should preserve resident output"),
                };
                assert_eq!(handle_precision(output), Some(ProviderPrecision::F32));
                assert_eq!(output.shape(), &[2, 2]);
                assert!(!output.shares_allocation_with(&input.handle));
                drop(rooted);
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
                if matches!(
                    mode,
                    Mode::RawShape
                        | Mode::RawInteger
                        | Mode::RawLogical
                        | Mode::RawOwner
                        | Mode::RawBorrowAlias
                ) {
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
