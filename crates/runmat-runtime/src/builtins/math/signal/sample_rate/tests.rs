use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, GpuTensorStorage,
    HostTensorOwned, HostTensorView, ProviderIirCoefficientSource, ProviderIirFilterOptions,
    ProviderIirFilterResult, ProviderPrecision,
};
use runmat_value::IntegerStorage;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;

struct HostileDownsampleProvider {
    owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    foreign_owner: runmat_accelerate_api::test_support::TestAllocationOwner,
    mode: AtomicU8,
    upload_mode: AtomicU8,
    numeric_uploads: AtomicUsize,
    staged_scatter: Mutex<Option<(GpuTensorHandle, GpuTensorHandle)>>,
    downloads: AtomicUsize,
    download_mode: AtomicU8,
    gathers: AtomicUsize,
    last_output: AtomicU64,
    pending_filter: AtomicBool,
    filter_output: AtomicU64,
    probes: Mutex<HashMap<u64, runmat_accelerate_api::test_support::RetirementProbe>>,
}

impl HostileDownsampleProvider {
    fn new() -> Self {
        Self {
            owner: Default::default(),
            foreign_owner: Default::default(),
            mode: AtomicU8::new(0),
            upload_mode: AtomicU8::new(0),
            numeric_uploads: AtomicUsize::new(0),
            staged_scatter: Mutex::new(None),
            downloads: AtomicUsize::new(0),
            download_mode: AtomicU8::new(0),
            gathers: AtomicUsize::new(0),
            last_output: AtomicU64::new(0),
            pending_filter: AtomicBool::new(false),
            filter_output: AtomicU64::new(0),
            probes: Default::default(),
        }
    }
    fn remember(
        &self,
        allocation: runmat_accelerate_api::test_support::TestAllocation,
    ) -> GpuTensorHandle {
        self.probes
            .lock()
            .unwrap()
            .insert(allocation.handle.buffer_id(), allocation.probe);
        allocation.handle
    }
    fn frees(&self) -> usize {
        self.probes
            .lock()
            .unwrap()
            .values()
            .map(|probe| probe.retirements())
            .sum()
    }
    fn is_live(&self, handle: &GpuTensorHandle) -> bool {
        self.probes
            .lock()
            .unwrap()
            .get(&handle.buffer_id())
            .is_some_and(|probe| probe.retirements() == 0)
    }
    fn live_count(&self) -> usize {
        self.probes
            .lock()
            .unwrap()
            .values()
            .filter(|probe| probe.retirements() == 0)
            .count()
    }
}

impl AccelProvider for HostileDownsampleProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.remember(self.owner.allocate_numeric(
            runmat_accelerate_api::HostNumericTensorOwned {
                data: runmat_accelerate_api::HostNumericDataOwned::F64(host.data.to_vec()),
                shape: host.shape.to_vec(),
                storage: GpuTensorStorage::Real,
            },
        )?))
    }
    fn upload_numeric_raw(
        &self,
        host: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.numeric_uploads.fetch_add(1, Ordering::Relaxed);
        let mode = self.upload_mode.load(Ordering::Relaxed);
        if mode == 7 {
            return Err(runmat_accelerate_api::ProviderExecutionError::from(
                runmat_accelerate_api::unsupported_provider_operation(
                    "coefficient upload declined",
                ),
            ));
        }
        if mode == 8 {
            return Err(runmat_accelerate_api::ProviderExecutionError::from_source(
                std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "coefficient upload cancelled",
                ),
            )
            .after_effect());
        }
        if mode == 4 || mode == 5 {
            let (source, padded) = self
                .staged_scatter
                .lock()
                .unwrap()
                .take()
                .expect("scatter staged");
            let borrowed = if mode == 4 { source } else { padded };
            return Ok(borrowed
                .with_view(runmat_accelerate_api::GpuTensorView::contiguous(
                    host.shape.to_vec(),
                ))
                .expect("borrowed upload view"));
        }
        let data = match host.data {
            runmat_accelerate_api::HostNumericDataView::F64(values) => values.to_vec(),
            _ => unreachable!("resample uploads F64 FIR coefficients"),
        };
        let shape = if mode == 1 {
            vec![1, data.len()]
        } else {
            host.shape.to_vec()
        };
        let payload = runmat_accelerate_api::HostNumericTensorOwned {
            data: if mode == 2 || mode == 6 {
                runmat_accelerate_api::HostNumericDataOwned::F32(
                    data.iter().map(|value| *value as f32).collect(),
                )
            } else {
                runmat_accelerate_api::HostNumericDataOwned::F64(data)
            },
            shape,
            storage: GpuTensorStorage::Real,
        };
        let owner = if mode == 3 {
            &self.foreign_owner
        } else {
            &self.owner
        };
        let allocation = owner
            .allocate_numeric(payload)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        if mode == 6 {
            allocation
                .probe
                .fail_retirement("injected coefficient retirement failure");
        }
        Ok(self.remember(allocation))
    }
    fn zeros_with_storage(
        &self,
        shape: &[usize],
        storage: GpuTensorStorage,
    ) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.remember(self.owner.allocate_zeroed(
            shape.to_vec(),
            runmat_accelerate_api::NumericElementType::F64,
            storage,
        )?))
    }
    fn scatter_linear(
        &self,
        target: &GpuTensorHandle,
        _indices: &[u32],
        values: &GpuTensorHandle,
    ) -> anyhow::Result<()> {
        *self.staged_scatter.lock().unwrap() = Some((values.clone(), target.clone()));
        Ok(())
    }
    fn iir_filter<'a>(
        &'a self,
        _coefficients: ProviderIirCoefficientSource<'a>,
        x: &'a GpuTensorHandle,
        _options: ProviderIirFilterOptions,
    ) -> AccelProviderFuture<'a, ProviderIirFilterResult> {
        if !self.pending_filter.load(Ordering::Relaxed) {
            return Box::pin(async {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "iir_filter not supported by provider",
                ))
            });
        }
        let output = self.remember(
            self.owner
                .allocate_zeroed(
                    x.shape().to_vec(),
                    runmat_accelerate_api::NumericElementType::F64,
                    GpuTensorStorage::Real,
                )
                .expect("pending filter output"),
        );
        self.filter_output
            .store(output.buffer_id(), Ordering::Relaxed);
        Box::pin(async move {
            std::future::pending::<()>().await;
            Ok(ProviderIirFilterResult {
                output,
                final_state: None,
            })
        })
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            match self.download_mode.load(Ordering::Relaxed) {
                1 => {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "scalar control download unavailable",
                    ))
                }
                2 => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "scalar control download interrupted",
                    )
                    .into())
                }
                _ => {}
            }
            let probe = self
                .probes
                .lock()
                .unwrap()
                .get(&handle.buffer_id())
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("unknown hostile test allocation"))?;
            let data = probe.with_data(handle, |data| match data {
                runmat_accelerate_api::HostNumericDataView::F64(data) => Ok(data.to_vec()),
                _ => Err(anyhow::anyhow!("test download expects F64")),
            })??;
            Ok(HostTensorOwned {
                data,
                shape: handle.shape().to_vec(),
                storage: handle.physical().storage(),
            })
        })
    }
    fn device_info(&self) -> String {
        "hostile-downsample-test-provider".to_string()
    }
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn gather_linear(
        &self,
        source: &GpuTensorHandle,
        _indices: &[u32],
        output_shape: &[usize],
    ) -> anyhow::Result<GpuTensorHandle> {
        self.gathers.fetch_add(1, Ordering::Relaxed);
        let mode = self.mode.load(Ordering::Relaxed);
        if mode == 6 {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "forced gather fallback",
            ));
        }
        if mode == 7 {
            return Ok(source.clone());
        }
        if mode == 8 {
            return Ok(
                source.with_view(runmat_accelerate_api::GpuTensorView::contiguous(
                    output_shape.to_vec(),
                ))?,
            );
        }
        let shape = if mode == 0 {
            vec![1, output_shape.iter().product::<usize>() + 1]
        } else {
            output_shape.to_vec()
        };
        let owner = if mode == 1 {
            &self.foreign_owner
        } else {
            &self.owner
        };
        let allocation = if mode == 5 {
            owner.allocate_logical_zeroed(shape, ProviderPrecision::F64)?
        } else {
            let storage = if mode == 2 {
                GpuTensorStorage::ComplexInterleaved
            } else {
                GpuTensorStorage::Real
            };
            let count = shape.iter().product::<usize>() * if mode == 2 { 2 } else { 1 };
            let data = match mode {
                3 => runmat_accelerate_api::HostNumericDataOwned::F32(vec![99.0; count]),
                4 => runmat_accelerate_api::HostNumericDataOwned::U8(vec![99; count]),
                _ => runmat_accelerate_api::HostNumericDataOwned::F64(vec![99.0; count]),
            };
            owner.allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
                data,
                shape,
                storage,
            })?
        };
        let output = self.remember(allocation);
        self.last_output
            .store(output.buffer_id(), Ordering::Relaxed);
        Ok(output)
    }
}

fn first_unrepresentable_usize_double() -> f64 {
    if usize::BITS == 64 {
        usize::MAX as f64
    } else {
        (usize::MAX as f64) + 1.0
    }
}

fn exact_platform_wide_integer() -> u64 {
    if usize::BITS == 64 {
        9_007_199_254_740_993
    } else {
        u32::MAX as u64
    }
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn call_upsample(args: Vec<Value>) -> Value {
    block_on(upsample_builtin(
        args[0].clone(),
        args[1].clone(),
        args[2..].to_vec(),
    ))
    .unwrap()
}

fn call_downsample(args: Vec<Value>) -> Value {
    block_on(downsample_builtin(
        args[0].clone(),
        args[1].clone(),
        args[2..].to_vec(),
    ))
    .unwrap()
}

fn call_resample(args: Vec<Value>) -> Value {
    block_on(resample_builtin(
        args[0].clone(),
        args[1].clone(),
        args[2].clone(),
        args[3..].to_vec(),
    ))
    .unwrap()
}

mod controls;
mod numerics;
mod precision;
mod provider_admission;
mod provider_faults;
mod cooperation;
mod cancellation;
mod discarded_outputs;
