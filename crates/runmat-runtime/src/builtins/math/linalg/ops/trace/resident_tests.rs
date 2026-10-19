//! Capable-provider contracts not exercised by the current production providers.
use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericDataOwned, HostNumericTensorOwned,
    HostTensorView, NumericElementType, ProviderExecutionFuture, ProviderPrecision,
    ThreadProviderGuard,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn trace_restore_failure_keeps_retry_span_context_and_typed_cause() {
    let source = build_runtime_error("provider upload failed")
        .with_identifier("RunMat:gpu:UploadFailed")
        .with_builtin("gpu upload")
        .with_task_id("task-7")
        .with_phase("restore")
        .with_span((11usize, 4usize).into())
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "device transfer failed",
        ))
        .build();
    let error = super::resident::trace_restore_failure(source);
    assert_eq!(error.identifier(), TRACE_ERROR_INTERNAL.identifier);
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(
        error.span.map(|span| (span.offset(), span.len())),
        Some((11, 4))
    );
    assert_eq!(error.context.builtin.as_deref(), Some("trace"));
    assert_eq!(error.context.task_id.as_deref(), Some("task-7"));
    assert_eq!(error.context.phase.as_deref(), Some("restore"));
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<std::io::Error>() {
        cause = cause.source().expect("restore source chain retained");
    }
    assert_eq!(
        cause
            .downcast_ref::<std::io::Error>()
            .map(std::io::Error::kind),
        Some(std::io::ErrorKind::BrokenPipe)
    );
}

struct ComplexTraceProvider {
    owner: TestAllocationOwner,
    diagonals: AtomicUsize,
    sums: AtomicUsize,
    casts: AtomicUsize,
    reject_cast: bool,
    downloads: AtomicUsize,
    values: Mutex<HashMap<u64, HostNumericTensorOwned>>,
}

impl ComplexTraceProvider {
    fn allocate(&self, shape: Vec<usize>, data: HostNumericDataOwned) -> GpuTensorHandle {
        let payload = HostNumericTensorOwned {
            data,
            shape,
            storage: GpuTensorStorage::ComplexInterleaved,
        };
        let handle = self
            .owner
            .allocate_numeric(payload.clone())
            .expect("fixture allocation")
            .handle;
        self.values
            .lock()
            .unwrap()
            .insert(handle.buffer_id(), payload);
        handle
    }

    fn payload(&self, handle: &GpuTensorHandle) -> anyhow::Result<HostNumericTensorOwned> {
        self.values
            .lock()
            .unwrap()
            .get(&handle.buffer_id())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fixture payload unavailable"))
    }
}

impl AccelProvider for ComplexTraceProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        // Preference is not proof that this owner lacks its F64 sum hook.
        ProviderPrecision::F32
    }

    fn device_info(&self) -> String {
        "complex trace contract fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.payload(handle)
        })
    }

    fn diag_extract(&self, matrix: &GpuTensorHandle, _: isize) -> anyhow::Result<GpuTensorHandle> {
        self.diagonals.fetch_add(1, Ordering::SeqCst);
        if matrix.shape() == [1, 1] {
            return Ok(matrix.clone());
        }
        let payload = self.payload(matrix)?;
        let rows = matrix.shape()[0];
        let diagonal = match payload.data {
            HostNumericDataOwned::F32(data) => HostNumericDataOwned::F32(
                (0..rows)
                    .flat_map(|index| {
                        let position = 2 * (index + index * rows);
                        [data[position], data[position + 1]]
                    })
                    .collect(),
            ),
            HostNumericDataOwned::F64(data) => HostNumericDataOwned::F64(
                (0..rows)
                    .flat_map(|index| {
                        let position = 2 * (index + index * rows);
                        [data[position], data[position + 1]]
                    })
                    .collect(),
            ),
            _ => anyhow::bail!("floating input required"),
        };
        Ok(self.allocate(vec![rows, 1], diagonal))
    }

    fn reduce_sum<'a>(
        &'a self,
        diagonal: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.sums.fetch_add(1, Ordering::SeqCst);
            if diagonal.shape() == [1, 1]
                && runmat_accelerate_api::handle_precision(diagonal) == Some(ProviderPrecision::F64)
            {
                return Ok(diagonal.clone());
            }
            let payload = self.payload(diagonal)?;
            let values: Vec<f64> = match payload.data {
                HostNumericDataOwned::F32(data) => data.into_iter().map(f64::from).collect(),
                HostNumericDataOwned::F64(data) => data,
                _ => return Err(anyhow::anyhow!("floating diagonal required").into()),
            };
            let (real, imaginary) = values
                .chunks_exact(2)
                .fold((0.0, 0.0), |sum, pair| (sum.0 + pair[0], sum.1 + pair[1]));
            Ok(self.allocate(vec![1, 1], HostNumericDataOwned::F64(vec![real, imaginary])))
        })
    }

    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let evaluate = || -> anyhow::Result<GpuTensorHandle> {
                self.casts.fetch_add(1, Ordering::SeqCst);
                if self.reject_cast {
                    return Err(runmat_accelerate_api::unsupported_provider_operation(
                        "fixture rejects Single adaptation",
                    ));
                }
                let payload = self.payload(input)?;
                let HostNumericDataOwned::F64(data) = payload.data else {
                    anyhow::bail!("double result required before single cast")
                };
                Ok(self.allocate(
                    vec![1, 1],
                    HostNumericDataOwned::F32(data.into_iter().map(|value| value as f32).collect()),
                ))
            };
            evaluate().map_err(Into::into)
        })
    }
}

#[test]
fn complex_diagonal_and_sum_remain_resident_for_capable_provider() {
    for element in [NumericElementType::F64, NumericElementType::F32] {
        let provider: &'static ComplexTraceProvider = Box::leak(Box::new(ComplexTraceProvider {
            owner: TestAllocationOwner::new(),
            diagonals: AtomicUsize::new(0),
            sums: AtomicUsize::new(0),
            casts: AtomicUsize::new(0),
            reject_cast: false,
            downloads: AtomicUsize::new(0),
            values: Mutex::new(HashMap::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let data = match element {
            NumericElementType::F32 => {
                HostNumericDataOwned::F32(vec![1.0, 2.0, 100.0, 200.0, -50.0, -60.0, 7.0, 8.0])
            }
            NumericElementType::F64 => {
                HostNumericDataOwned::F64(vec![1.0, 2.0, 100.0, 200.0, -50.0, -60.0, 7.0, 8.0])
            }
            _ => unreachable!(),
        };
        let input = provider.allocate(vec![2, 2], data);
        assert_eq!(
            provider.preflight_double_sum(&input),
            runmat_accelerate_api::ProviderDoubleSumPreflight::Unknown,
        );
        let output = block_on(super::trace_builtin(Value::GpuTensor(input.clone()))).unwrap();
        let Value::GpuTensor(output) = output else {
            panic!("capable provider must keep complex trace resident")
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            runmat_accelerate_api::handle_precision(&input)
        );
        assert_eq!(provider.diagonals.load(Ordering::SeqCst), 1);
        assert_eq!(provider.sums.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.casts.load(Ordering::SeqCst),
            usize::from(element == NumericElementType::F32)
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
            .expect("gather capable-provider trace result");
        let Value::ComplexTensor(gathered) = gathered else {
            panic!("trace must preserve complex storage");
        };
        assert_eq!(gathered.materialize_f64(), vec![(8.0, 10.0)]);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        input.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn completed_trace_reduction_does_not_replay_after_single_cast_unsupported() {
    let provider: &'static ComplexTraceProvider = Box::leak(Box::new(ComplexTraceProvider {
        owner: TestAllocationOwner::new(),
        diagonals: AtomicUsize::new(0),
        sums: AtomicUsize::new(0),
        casts: AtomicUsize::new(0),
        reject_cast: true,
        downloads: AtomicUsize::new(0),
        values: Mutex::new(HashMap::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocate(
        vec![2, 2],
        HostNumericDataOwned::F32(vec![1.0, 2.0, 100.0, 200.0, -50.0, -60.0, 7.0, 8.0]),
    );
    let error = block_on(super::trace_builtin(Value::GpuTensor(input.clone())))
        .expect_err("post-reduction Unsupported must remain terminal");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.source.is_some(), "typed provider cause must survive");
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>() {
        cause = cause
            .source()
            .expect("original typed cast rejection retained");
    }
    assert_eq!(provider.diagonals.load(Ordering::SeqCst), 1);
    assert_eq!(provider.sums.load(Ordering::SeqCst), 1);
    assert_eq!(provider.casts.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    input.allocation_lease().validate_live().unwrap();
}

#[test]
fn scalar_complex_diagonal_and_sum_may_keep_the_input_allocation() {
    let provider: &'static ComplexTraceProvider = Box::leak(Box::new(ComplexTraceProvider {
        owner: TestAllocationOwner::new(),
        diagonals: AtomicUsize::new(0),
        sums: AtomicUsize::new(0),
        casts: AtomicUsize::new(0),
        reject_cast: false,
        downloads: AtomicUsize::new(0),
        values: Mutex::new(HashMap::new()),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.allocate(vec![1, 1], HostNumericDataOwned::F64(vec![3.0, -4.0]));
    let output = block_on(super::trace_builtin(Value::GpuTensor(input.clone()))).unwrap();
    let Value::GpuTensor(output) = output else {
        panic!("scalar complex trace must stay resident")
    };
    assert!(output.shares_allocation_with(&input));
    assert_eq!(provider.diagonals.load(Ordering::SeqCst), 1);
    assert_eq!(provider.sums.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
        .expect("gather scalar trace result");
    let Value::ComplexTensor(gathered) = gathered else {
        panic!("scalar trace must retain complex storage");
    };
    assert_eq!(gathered.materialize_f64(), vec![(3.0, -4.0)]);
    input.allocation_lease().validate_live().unwrap();
}
