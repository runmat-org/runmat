#![cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    *,
};
use runmat_runtime::GpuGatherRetry;
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug)]
enum Fault {
    Shape,
    Precision,
    Storage,
    Integer,
    Logical,
    Foreign,
    Unavailable,
    CurrentAlias,
    OriginalAlias,
    None,
}
struct Provider {
    owner: TestAllocationOwner,
    foreign: TestAllocationOwner,
    fault: Fault,
    fault_pass: usize,
    calls: AtomicUsize,
    original: Mutex<Option<GpuTensorHandle>>,
    outputs: Mutex<Vec<RetirementProbe>>,
    input_intents: Mutex<Vec<Option<GpuHandleProvenance>>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "forward FFT output admission".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no replay/upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("no fallback/download")
    }
    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        panic!("no fallback/native download")
    }
    fn fft_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        length: Option<usize>,
        axis: usize,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.input_intents
                .lock()
                .unwrap()
                .push(handle_provenance(input));
            let pass = self.calls.fetch_add(1, Ordering::SeqCst);
            let fault = if pass == self.fault_pass {
                self.fault
            } else {
                Fault::None
            };
            let mut shape = input.shape().to_vec();
            while shape.len() <= axis {
                shape.push(1);
            }
            if let Some(length) = length {
                shape[axis] = length;
            }
            let alias = match fault {
                Fault::CurrentAlias => Some(input.clone()),
                Fault::OriginalAlias => self.original.lock().unwrap().clone(),
                _ => None,
            };
            if let Some(alias) = alias {
                return Ok(alias
                    .with_view(alias.view().clone().with_shape(shape))
                    .unwrap());
            }
            if matches!(fault, Fault::Shape) {
                shape[0] += 1;
            }
            let owner = if matches!(fault, Fault::Foreign) {
                &self.foreign
            } else {
                &self.owner
            };
            let allocation = if matches!(fault, Fault::Logical) {
                owner.allocate_logical_zeroed(shape, ProviderPrecision::F64)?
            } else {
                owner.allocate_zeroed(
                    shape,
                    match fault {
                        Fault::Precision => NumericElementType::F32,
                        Fault::Integer => NumericElementType::I32,
                        _ if handle_precision(input) == Some(ProviderPrecision::F32) => {
                            NumericElementType::F32
                        }
                        _ => NumericElementType::F64,
                    },
                    if matches!(fault, Fault::Storage) {
                        GpuTensorStorage::Real
                    } else {
                        GpuTensorStorage::ComplexInterleaved
                    },
                )?
            };
            if matches!(fault, Fault::Unavailable) {
                allocation.probe.mark_unavailable();
            }
            self.outputs.lock().unwrap().push(allocation.probe);
            Ok(allocation.handle)
        })
    }
}
fn provider(fault: Fault, fault_pass: usize) -> &'static Provider {
    Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        foreign: TestAllocationOwner::new(),
        fault,
        fault_pass,
        calls: AtomicUsize::new(0),
        original: Mutex::new(None),
        outputs: Mutex::new(Vec::new()),
        input_intents: Mutex::new(Vec::new()),
    }))
}

#[test]
fn forward_passes_preserve_explicit_and_automatic_intent_without_mutating_inputs() {
    for intent in [
        GpuHandleProvenance::Explicit,
        GpuHandleProvenance::Automatic,
    ] {
        for (name, fault, pass, shape, count) in [
            ("fft", Fault::None, 0, vec![2, 3], 1),
            ("fft2", Fault::None, 0, vec![2, 3], 2),
            ("fftn", Fault::None, 0, vec![2, 3], 2),
            ("fft", Fault::CurrentAlias, 0, vec![1, 1], 1),
            ("fft2", Fault::CurrentAlias, 1, vec![2, 1], 2),
            ("fftn", Fault::CurrentAlias, 1, vec![2, 1], 2),
        ] {
            let provider = provider(fault, pass);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(
                    shape,
                    NumericElementType::F64,
                    GpuTensorStorage::ComplexInterleaved,
                )
                .unwrap();
            let source = input.handle.with_provenance(intent).unwrap();
            let original_intent = handle_provenance(&input.handle);
            let args = [Value::GpuTensor(source.clone())];
            let result = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
            let Value::GpuTensor(output) = result.value() else {
                panic!("resident result")
            };
            assert_eq!(handle_provenance(output), Some(intent), "{name} {fault:?}");
            assert_eq!(
                *provider.input_intents.lock().unwrap(),
                vec![Some(intent); count]
            );
            assert_eq!(handle_provenance(&source), Some(intent));
            assert_eq!(handle_provenance(&input.handle), original_intent);
            assert_eq!(provider.calls.load(Ordering::SeqCst), count);
            drop(result);
            drop(args);
            drop(source);
            assert_eq!(input.probe.retirements(), 0);
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
            for output in provider.outputs.lock().unwrap().iter() {
                assert_eq!(output.retirements(), 1);
            }
        }
    }
}

#[test]
fn malformed_forward_outputs_stop_before_publication_or_next_pass() {
    for name in ["fft", "fft2", "fftn"] {
        for fault in [
            Fault::Shape,
            Fault::Precision,
            Fault::Storage,
            Fault::Integer,
            Fault::Logical,
            Fault::Foreign,
            Fault::Unavailable,
            Fault::CurrentAlias,
            Fault::OriginalAlias,
        ] {
            for pass in 0..if name == "fft" { 1 } else { 2 } {
                let provider = provider(fault, pass);
                let _selected = ThreadProviderGuard::set(Some(provider));
                let input = provider
                    .owner
                    .allocate_zeroed(
                        vec![2, 2],
                        NumericElementType::F64,
                        GpuTensorStorage::ComplexInterleaved,
                    )
                    .unwrap();
                *provider.original.lock().unwrap() = Some(input.handle.clone());
                let args = [Value::GpuTensor(input.handle.clone())];
                let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
                assert_eq!(
                    error.gpu_gather_retry,
                    GpuGatherRetry::Never,
                    "{name} {fault:?}: {error}"
                );
                assert_eq!(
                    error.identifier(),
                    Some(format!("RunMat:{name}:Internal").as_str())
                );
                assert_eq!(
                    provider.calls.load(Ordering::SeqCst),
                    pass + 1,
                    "{name} {fault:?}"
                );
                for output in provider.outputs.lock().unwrap().iter() {
                    assert_eq!(
                        output.retirements(),
                        1,
                        "{name} {fault:?}: rejected/intermediate output"
                    );
                }
                assert_eq!(input.probe.retirements(), 0);
                input.handle.allocation_lease().validate_live().unwrap();
                drop(error);
                drop(args);
                provider.original.lock().unwrap().take();
                drop(input.handle);
                assert_eq!(input.probe.retirements(), 1);
            }
        }
    }
}

#[test]
fn unavailable_input_never_reaches_a_provider_hook() {
    for name in ["fft", "fft2", "fftn"] {
        let provider = provider(Fault::None, 0);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        input.probe.mark_unavailable();
        let args = [Value::GpuTensor(input.handle.clone())];
        let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
        drop(error);
        drop(args);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[test]
fn unchanged_complex_unit_axis_may_return_an_immutable_view_alias() {
    for length in [None, Some(1)] {
        let provider = provider(Fault::CurrentAlias, 0);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.25, -2.0, 3.5, 4.0]),
                shape: vec![1, 2],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap();
        let args = vec![
            Value::GpuTensor(input.handle.clone()),
            length.map_or_else(
                || Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                |n| Value::Num(n as f64),
            ),
            Value::Num(1.0),
        ];
        let output = block_on(runmat_runtime::call_builtin_async("fft", &args)).unwrap();
        let Value::GpuTensor(resident_output) = output.value() else {
            panic!("resident no-op remains resident")
        };
        assert!(resident_output.shares_allocation_with(&input.handle));
        assert_ne!(
            resident_output.view_identity(),
            input.handle.view_identity()
        );
        assert_eq!(resident_output.shape(), &[1, 2]);
        input
            .probe
            .with_data(&input.handle, |data| {
                let HostNumericDataView::F64(values) = data else {
                    panic!("native double")
                };
                assert_eq!(values, &[1.25, -2.0, 3.5, 4.0]);
            })
            .unwrap();
        drop(output);
        drop(args);
        assert_eq!(input.probe.retirements(), 0);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}

#[test]
fn second_noop_pass_can_alias_current_but_cannot_resurrect_original_input() {
    for name in ["fft2", "fftn"] {
        for fault in [Fault::CurrentAlias, Fault::OriginalAlias] {
            let provider = provider(fault, 1);
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(
                    vec![2, 1],
                    NumericElementType::F64,
                    GpuTensorStorage::ComplexInterleaved,
                )
                .unwrap();
            *provider.original.lock().unwrap() = Some(input.handle.clone());
            let args = [Value::GpuTensor(input.handle.clone())];
            let result = block_on(runmat_runtime::call_builtin_async(name, &args));
            match fault {
                Fault::CurrentAlias => {
                    let result = result.unwrap();
                    let Value::GpuTensor(output) = result.value() else {
                        panic!("resident result")
                    };
                    assert!(!output.shares_allocation_with(&input.handle));
                    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
                    drop(result);
                }
                Fault::OriginalAlias => {
                    assert_eq!(result.unwrap_err().gpu_gather_retry, GpuGatherRetry::Never)
                }
                _ => unreachable!(),
            }
            assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
            assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
            assert_eq!(input.probe.retirements(), 0);
            drop(args);
            provider.original.lock().unwrap().take();
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[test]
fn native_single_output_is_admitted_independently_of_preferred_double_precision() {
    for name in ["fft", "fft2", "fftn"] {
        let provider = provider(Fault::None, 0);
        let _selected = ThreadProviderGuard::set(Some(provider));
        assert_eq!(provider.precision(), ProviderPrecision::F64);
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let args = [Value::GpuTensor(input.handle.clone())];
        let result = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("native resident output")
        };
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(handle_precision(output), Some(ProviderPrecision::F32));
        assert_eq!(handle_storage(output), GpuTensorStorage::ComplexInterleaved);
        drop(result);
        drop(args);
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
        for output in provider.outputs.lock().unwrap().iter() {
            assert_eq!(output.retirements(), 1);
        }
    }
}
