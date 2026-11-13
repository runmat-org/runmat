//! The shared circshift admission decision precedes all provider effects.
#![cfg(not(target_arch = "wasm32"))]

use futures::executor::block_on;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::*;
use runmat_runtime::GpuGatherRetry;
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum Mode {
    Unavailable,
    Success,
    Failure,
    Alias,
}

fn owned_numeric(data: HostNumericDataView<'_>) -> HostNumericDataOwned {
    match data {
        HostNumericDataView::F64(values) => HostNumericDataOwned::F64(values.to_vec()),
        HostNumericDataView::F32(values) => HostNumericDataOwned::F32(values.to_vec()),
        HostNumericDataView::I8(values) => HostNumericDataOwned::I8(values.to_vec()),
        HostNumericDataView::I16(values) => HostNumericDataOwned::I16(values.to_vec()),
        HostNumericDataView::I32(values) => HostNumericDataOwned::I32(values.to_vec()),
        HostNumericDataView::I64(values) => HostNumericDataOwned::I64(values.to_vec()),
        HostNumericDataView::U8(values) => HostNumericDataOwned::U8(values.to_vec()),
        HostNumericDataView::U16(values) => HostNumericDataOwned::U16(values.to_vec()),
        HostNumericDataView::U32(values) => HostNumericDataOwned::U32(values.to_vec()),
        HostNumericDataView::U64(values) => HostNumericDataOwned::U64(values.to_vec()),
    }
}

struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    source: Mutex<Option<RetirementProbe>>,
    probes: Mutex<Vec<RetirementProbe>>,
    calls: AtomicUsize,
    downloads: AtomicUsize,
    uploads: AtomicUsize,
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "circshift preflight fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("native transfer must use its typed contract")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
            data: owned_numeric(host.data),
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        let allocation = self
            .owner
            .allocate_logical(host.data.to_vec(), host.shape.to_vec())
            .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
        self.probes.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("native download must use its typed contract")
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.source
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .with_data(handle, |data| HostNumericTensorOwned {
                    data: owned_numeric(data),
                    shape: handle.shape().to_vec(),
                    storage: handle_storage(handle),
                })
        })
    }
    fn supports_circshift(&self, _: &GpuTensorHandle, _: &[isize]) -> bool {
        !matches!(self.mode, Mode::Unavailable)
    }
    fn reshape(&self, _: &GpuTensorHandle, _: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        panic!("circshift admission must not reshape a borrowed input")
    }
    fn circshift(
        &self,
        input: &GpuTensorHandle,
        shifts: &[isize],
    ) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::Unavailable => panic!("unavailable hook was invoked"),
            Mode::Success => {
                let data =
                    self.source
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .with_data(input, |data| {
                            let HostNumericDataView::F64(values) = data else {
                                panic!("fixture input is F64")
                            };
                            let mut values = values.to_vec();
                            let shift = shifts[0].rem_euclid(values.len() as isize) as usize;
                            values.rotate_right(shift);
                            values
                        })?;
                let mut shape = input.shape().to_vec();
                shape.resize(shape.len().max(shifts.len()), 1);
                let allocation = self.owner.allocate_numeric(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(data),
                    shape,
                    storage: handle_storage(input),
                })?;
                self.probes.lock().unwrap().push(allocation.probe);
                Ok(allocation.handle)
            }
            Mode::Failure => anyhow::bail!("circshift not supported by provider"),
            Mode::Alias => Ok(input.clone()),
        }
    }
}

fn exercise(name: &str, mode: Mode) {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        mode,
        source: Mutex::new(None),
        probes: Mutex::new(Vec::new()),
        calls: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        uploads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![4, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    *provider.source.lock().unwrap() = Some(input.probe.clone());
    let mut args = vec![Value::GpuTensor(input.handle.clone())];
    if name == "circshift" {
        args.push(Value::Num(1.0));
    }
    match mode {
        Mode::Failure | Mode::Alias => {
            let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
            assert_eq!(
                error.gpu_gather_retry,
                GpuGatherRetry::Never,
                "{name}: {error}"
            );
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1, "{name}");
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0, "{name}");
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0, "{name}");
            if matches!(mode, Mode::Failure) {
                assert!(error
                    .to_string()
                    .contains("circshift not supported by provider"));
            } else {
                assert!(error.to_string().contains("aliased its input"));
                assert_eq!(input.probe.retirements(), 0, "borrowed input was retired");
            }
        }
        Mode::Unavailable => {
            let _ = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
            assert_eq!(provider.calls.load(Ordering::SeqCst), 0, "{name}");
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1, "{name}");
        }
        Mode::Success => {
            let result = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
            let Value::GpuTensor(output) = result.value() else {
                panic!("{name}: native result lost residency")
            };
            assert_eq!(output.shape(), &[4, 1]);
            assert!(!output.shares_allocation_with(&input.handle));
            let actual = provider
                .probes
                .lock()
                .unwrap()
                .last()
                .unwrap()
                .with_data(output, |data| {
                    let HostNumericDataView::F64(values) = data else {
                        panic!("fixture result is F64")
                    };
                    values.to_vec()
                })
                .unwrap();
            let expected = if name == "circshift" {
                vec![4.0, 1.0, 2.0, 3.0]
            } else {
                vec![3.0, 4.0, 1.0, 2.0]
            };
            assert_eq!(actual, expected, "{name}");
            assert_eq!(provider.calls.load(Ordering::SeqCst), 1, "{name}");
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0, "{name}");
        }
    }
    input.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn fft_and_array_shifts_share_preflight_and_never_replay_admitted_failures() {
    for name in ["fftshift", "ifftshift", "circshift"] {
        exercise(name, Mode::Unavailable);
        exercise(name, Mode::Success);
        exercise(name, Mode::Failure);
    }
    exercise("circshift", Mode::Alias);
}

#[test]
fn unavailable_circshift_preserves_native_floating_integer_logical_and_complex_classes() {
    let cases = [
        (
            HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0]),
            GpuTensorStorage::Real,
            false,
            HostNumericDataOwned::F64(vec![4.0, 1.0, 2.0, 3.0]),
        ),
        (
            HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0]),
            GpuTensorStorage::Real,
            false,
            HostNumericDataOwned::F32(vec![4.0, 1.0, 2.0, 3.0]),
        ),
        (
            HostNumericDataOwned::I64(vec![1, 2, 3, 4]),
            GpuTensorStorage::Real,
            false,
            HostNumericDataOwned::I64(vec![4, 1, 2, 3]),
        ),
        (
            HostNumericDataOwned::U8(vec![1, 0, 1, 0]),
            GpuTensorStorage::Real,
            true,
            HostNumericDataOwned::U8(vec![0, 1, 0, 1]),
        ),
        (
            HostNumericDataOwned::F64(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]),
            GpuTensorStorage::ComplexInterleaved,
            false,
            HostNumericDataOwned::F64(vec![4.0, 40.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0]),
        ),
    ];
    for (data, storage, logical, expected) in cases {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            mode: Mode::Unavailable,
            source: Mutex::new(None),
            probes: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = if logical {
            provider
                .owner
                .allocate_logical(vec![1, 0, 1, 0], vec![4, 1])
                .unwrap()
        } else {
            provider
                .owner
                .allocate_numeric(HostNumericTensorOwned {
                    data,
                    shape: vec![4, 1],
                    storage,
                })
                .unwrap()
        };
        *provider.source.lock().unwrap() = Some(input.probe.clone());
        let descriptor = input.handle.descriptor();
        let args = [Value::GpuTensor(input.handle.clone()), Value::Num(1.0)];
        let result = block_on(runmat_runtime::call_builtin_async("circshift", &args)).unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("exact class lost residency")
        };
        assert_eq!(output.shape(), &[4, 1]);
        assert_eq!(handle_storage(output), storage);
        assert_eq!(handle_is_logical(output), logical);
        assert!(!output.shares_allocation_with(&input.handle));
        let actual = provider
            .probes
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .with_data(output, owned_numeric)
            .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(input.handle.descriptor(), descriptor);
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[test]
fn extended_rank_admitted_failure_preserves_original_source_metadata() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        mode: Mode::Failure,
        source: Mutex::new(None),
        probes: Mutex::new(Vec::new()),
        calls: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        uploads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![4, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    *provider.source.lock().unwrap() = Some(input.probe.clone());
    let before = input.handle.descriptor();
    let shift = runmat_value::Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
    let args = [Value::GpuTensor(input.handle.clone()), Value::Tensor(shift)];
    let error = block_on(runmat_runtime::call_builtin_async("circshift", &args)).unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(input.handle.descriptor(), before);
    assert_eq!(input.handle.shape(), &[4, 1]);
    input.handle.allocation_lease().validate_live().unwrap();
}

fn assert_native_noop(provider: &'static dyn AccelProvider) {
    let _selected = ThreadProviderGuard::set(Some(provider));
    for (values, shape, shift) in [
        (vec![7.0], vec![1, 1], 1.0),
        (Vec::new(), vec![0, 1], 1.0),
        (vec![1.0, 2.0, 3.0, 4.0], vec![4, 1], 4.0),
    ] {
        let input = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .unwrap();
        for name in ["circshift", "fftshift", "ifftshift"] {
            if name != "circshift" && values.len() > 1 {
                continue;
            }
            let mut args = vec![Value::GpuTensor(input.clone())];
            if name == "circshift" {
                args.push(Value::Num(shift));
            }
            let result = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
            let Value::GpuTensor(output) = result.value() else {
                panic!("{name}: no-op lost residency")
            };
            assert!(
                output.shares_allocation_with(&input),
                "{name}: no-op allocated or gathered"
            );
            assert_eq!(output.shape(), shape, "{name}");
            input.allocation_lease().validate_live().unwrap();
        }
    }
}

#[test]
fn actual_simple_provider_keeps_scalar_empty_and_modulo_zero_shifts_resident() {
    let provider = Box::leak(Box::new(
        runmat_accelerate::simple_provider::InProcessProvider::new(),
    ));
    assert_native_noop(provider);
}

#[test]
fn actual_simple_complex_shifts_use_exact_owned_fallback_for_all_three_consumers() {
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]),
            shape: &[4, 1],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .expect("native complex upload");
    let original = input.descriptor();
    assert!(!provider.supports_circshift(&input, &[1, 0]));
    for (name, expected) in [
        ("circshift", [4.0, 40.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0]),
        ("fftshift", [3.0, 30.0, 4.0, 40.0, 1.0, 10.0, 2.0, 20.0]),
        ("ifftshift", [3.0, 30.0, 4.0, 40.0, 1.0, 10.0, 2.0, 20.0]),
    ] {
        let mut args = vec![Value::GpuTensor(input.clone())];
        if name == "circshift" {
            args.push(Value::Num(1.0));
        }
        let result = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("{name}: complex class lost residency")
        };
        assert_eq!(output.shape(), &[4, 1], "{name}");
        assert_eq!(
            handle_storage(output),
            GpuTensorStorage::ComplexInterleaved,
            "{name}"
        );
        assert!(!output.shares_allocation_with(&input), "{name}");
        assert!(
            provider_for_handle(output)
                .is_some_and(|owner| same_provider_instance(owner, provider)),
            "{name}"
        );
        let downloaded = block_on(provider.download_numeric(output)).unwrap();
        let HostNumericDataOwned::F64(actual) = downloaded.data else {
            panic!("{name}: complex precision changed")
        };
        assert_eq!(actual, expected, "{name}");
        assert_eq!(
            input.descriptor(),
            original,
            "{name}: input metadata changed"
        );
        input.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn actual_wgpu_provider_keeps_scalar_empty_and_modulo_zero_shifts_resident() {
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("this actual WGPU witness requires an available adapter");
    assert_native_noop(provider);
}
