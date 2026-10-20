use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorStorage, HostTensorOwned, HostTensorView,
    NumericElementType, ProviderExecutionError, ProviderExecutionFuture, ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Mode {
    Unsupported,
    Execution,
    UnprovenUnsupported,
    AfterEffectUnsupported,
    Success,
    Pending,
    WrongOwner,
    Stale,
    WrongShape,
    WrongShapeCleanupFailure,
    WrongClass,
    WrongStorage,
    Alias,
    SquareSuccess,
    SquareAlias,
    SquareWrongShape,
    SquareErrorCleanupFailure,
    SquareSuccessCleanupFailure,
}

struct StdOwner {
    owner: TestAllocationOwner,
    foreign_owner: TestAllocationOwner,
    mode: Mode,
    element: NumericElementType,
    output_shape: Vec<usize>,
    calls: AtomicUsize,
    downloads: AtomicUsize,
    squares: AtomicUsize,
    outputs: Mutex<Vec<RetirementProbe>>,
    options: Mutex<Vec<(Option<usize>, ProviderStdNormalization, ProviderNanMode)>>,
}

impl StdOwner {
    fn fixture(mode: Mode) -> (&'static Self, GpuTensorHandle, RetirementProbe) {
        Self::fixture_with(mode, NumericElementType::F64, vec![3, 1], vec![1, 1])
    }

    fn fixture_with(
        mode: Mode,
        element: NumericElementType,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> (&'static Self, GpuTensorHandle, RetirementProbe) {
        let provider = Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            foreign_owner: TestAllocationOwner::new(),
            mode,
            element,
            output_shape,
            calls: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            squares: AtomicUsize::new(0),
            outputs: Mutex::new(Vec::new()),
            options: Mutex::new(Vec::new()),
        }));
        let input = provider
            .owner
            .allocate_zeroed(input_shape, element, GpuTensorStorage::Real)
            .unwrap();
        (provider, input.handle, input.probe)
    }

    async fn reduce(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let unsupported =
            || runmat_accelerate_api::unsupported_provider_operation("fixture std unavailable");
        match self.mode {
            Mode::Unsupported => Err(ProviderExecutionError::from(unsupported())),
            Mode::Execution => Err(ProviderExecutionError::from_source(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "std driver failed",
            ))),
            Mode::UnprovenUnsupported => {
                Err(ProviderExecutionError::from(unsupported()).with_unproven_effects())
            }
            Mode::AfterEffectUnsupported => {
                Err(ProviderExecutionError::from(unsupported()).after_effect())
            }
            Mode::Pending => std::future::pending().await,
            Mode::Alias => Ok(input.clone()),
            Mode::Success
            | Mode::WrongOwner
            | Mode::Stale
            | Mode::WrongShape
            | Mode::WrongShapeCleanupFailure
            | Mode::WrongClass
            | Mode::WrongStorage
            | Mode::SquareSuccess
            | Mode::SquareAlias
            | Mode::SquareWrongShape
            | Mode::SquareErrorCleanupFailure
            | Mode::SquareSuccessCleanupFailure => {
                let owner = if matches!(self.mode, Mode::WrongOwner) {
                    &self.foreign_owner
                } else {
                    &self.owner
                };
                let shape =
                    if matches!(self.mode, Mode::WrongShape | Mode::WrongShapeCleanupFailure) {
                        vec![2, 1]
                    } else {
                        self.output_shape.clone()
                    };
                let element = if matches!(self.mode, Mode::WrongClass) {
                    NumericElementType::F32
                } else {
                    self.element
                };
                let storage = if matches!(self.mode, Mode::WrongStorage) {
                    GpuTensorStorage::ComplexInterleaved
                } else {
                    GpuTensorStorage::Real
                };
                let output = owner
                    .allocate_zeroed(shape, element, storage)
                    .map_err(|error| ProviderExecutionError::from(error).with_unproven_effects())?;
                if matches!(self.mode, Mode::Stale) {
                    output.probe.mark_unavailable();
                }
                if matches!(self.mode, Mode::WrongShapeCleanupFailure) {
                    output.probe.fail_retirement("fixture cleanup failed");
                }
                if matches!(
                    self.mode,
                    Mode::SquareErrorCleanupFailure | Mode::SquareSuccessCleanupFailure
                ) {
                    output
                        .probe
                        .fail_retirement("std intermediate retirement failed");
                }
                self.outputs.lock().unwrap().push(output.probe);
                Ok(output.handle)
            }
        }
    }
}

impl AccelProvider for StdOwner {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "typed std phase fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        panic!("host restoration is not part of this witness")
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            handle.allocation_lease().validate_live()?;
            Ok(HostTensorOwned {
                data: vec![1.0; handle.shape().iter().product()],
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn reduce_std<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.options
            .lock()
            .unwrap()
            .push((None, normalization, nan_mode));
        Box::pin(self.reduce(input))
    }
    fn reduce_std_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
        normalization: ProviderStdNormalization,
        nan_mode: ProviderNanMode,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.options
            .lock()
            .unwrap()
            .push((Some(dim), normalization, nan_mode));
        Box::pin(self.reduce(input))
    }
    fn elem_mul<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.squares.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if matches!(self.mode, Mode::SquareAlias) {
                return Ok(left.clone());
            }
            if matches!(
                self.mode,
                Mode::SquareSuccess | Mode::SquareWrongShape | Mode::SquareSuccessCleanupFailure
            ) {
                let shape = if matches!(self.mode, Mode::SquareWrongShape) {
                    vec![2, 1]
                } else {
                    self.output_shape.clone()
                };
                let output = self
                    .owner
                    .allocate_zeroed(shape, self.element, GpuTensorStorage::Real)
                    .map_err(|error| ProviderExecutionError::from(error).after_effect())?;
                self.outputs.lock().unwrap().push(output.probe);
                return Ok(output.handle);
            }
            Err(ProviderExecutionError::from_source(std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "variance square failed",
            )))
        })
    }
}

fn all_axes() -> ParsedArguments {
    ParsedArguments {
        axes: StdAxes::All,
        normalization: StdNormalization::Sample,
        nan_mode: ReductionNaN::Include,
    }
}

fn retained_provider_cause(error: &crate::RuntimeError) -> &ProviderExecutionError {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(typed) = source.downcast_ref::<ProviderExecutionError>() {
            return typed;
        }
        source = source.source().expect("original provider cause");
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn std_initial_hook_falls_back_only_for_typed_pre_effect_unsupported() {
    for mode in [
        Mode::Unsupported,
        Mode::Execution,
        Mode::UnprovenUnsupported,
        Mode::AfterEffectUnsupported,
    ] {
        let (provider, handle, probe) = StdOwner::fixture(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let result = futures::executor::block_on(std_gpu(handle.clone(), &all_axes()));
        if matches!(mode, Mode::Unsupported) {
            result.expect("certified unsupported may gather once");
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        } else {
            let error = result.expect_err("non-fallback initial std failure");
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error.message().contains("reduce_std"));
            let cause = retained_provider_cause(&error);
            assert!(!cause.is_unsupported());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert!(provider.outputs.lock().unwrap().is_empty());
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn var_dim_std_failure_is_terminal_and_successful_std_makes_square_terminal() {
    for mode in [
        Mode::Unsupported,
        Mode::Execution,
        Mode::UnprovenUnsupported,
        Mode::AfterEffectUnsupported,
        Mode::Success,
    ] {
        let (provider, handle, probe) = StdOwner::fixture(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let result = futures::executor::block_on(super::super::var::var_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ));
        if matches!(mode, Mode::Unsupported) {
            result.expect("certified initial unsupported may gather");
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        } else {
            let error = result.expect_err("std or later square is terminal");
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            let cause = retained_provider_cause(&error);
            if matches!(mode, Mode::Success) {
                assert!(cause.is_after_effect());
                assert!(error.message().contains("square"));
                assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
            } else {
                assert!(!cause.is_unsupported());
                assert!(error.message().contains("reduce_std_dim"));
            }
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_std_future_does_not_gather_or_retire_borrowed_input_on_drop() {
    let (provider, handle, probe) = StdOwner::fixture(Mode::Pending);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let arguments = all_axes();
    let mut future = Box::pin(std_gpu(handle.clone(), &arguments));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    drop(future);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    handle.allocation_lease().validate_live().unwrap();
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn successful_std_stays_resident_with_exact_owner() {
    let (provider, handle, probe) = StdOwner::fixture(Mode::Success);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let output = futures::executor::block_on(std_gpu(handle.clone(), &all_axes()))
        .expect("resident std succeeds");
    let Value::GpuTensor(result) = output else {
        panic!("successful std must remain resident");
    };
    result.allocation_lease().validate_live().unwrap();
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_std_success_is_terminal_and_retires_only_the_rejected_result() {
    for mode in [
        Mode::WrongOwner,
        Mode::Stale,
        Mode::WrongShape,
        Mode::WrongClass,
        Mode::WrongStorage,
        Mode::Alias,
    ] {
        let (provider, handle, input_probe) = StdOwner::fixture(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let error = futures::executor::block_on(std_gpu(handle.clone(), &all_axes()))
            .expect_err("malformed successful reduction is terminal");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error.message().contains("invalid resident reduction"));
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.squares.load(Ordering::SeqCst), 0);
        let outputs = provider.outputs.lock().unwrap();
        if matches!(mode, Mode::Alias) {
            assert!(outputs.is_empty());
        } else {
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].retirements(), 1);
        }
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input_probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn variance_square_error_preserves_primary_when_std_retirement_also_fails() {
    let (provider, handle, input_probe) = StdOwner::fixture(Mode::SquareErrorCleanupFailure);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let error = futures::executor::block_on(super::super::var::var_builtin(
        Value::GpuTensor(handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("square and intermediate cleanup both fail");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().contains("resident variance square failed"));
    assert!(retained_provider_cause(&error).is_after_effect());
    let combined = std::error::Error::source(&error).expect("combined failure source");
    assert!(combined.to_string().contains("cleanup also failed"));
    assert!(combined
        .to_string()
        .contains("std intermediate retirement failed"));
    let outputs = provider.outputs.lock().unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].retirements(), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input_probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn successful_square_is_retired_if_std_intermediate_retirement_fails() {
    let (provider, handle, input_probe) = StdOwner::fixture(Mode::SquareSuccessCleanupFailure);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let error = futures::executor::block_on(super::super::var::var_builtin(
        Value::GpuTensor(handle.clone()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("successful square cannot publish after intermediate cleanup failure");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error
        .message()
        .contains("std intermediate retirement failed"));
    let outputs = provider.outputs.lock().unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].retirements(), 1);
    assert_eq!(outputs[1].retirements(), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input_probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn var_rejects_invalid_std_before_square_and_keeps_borrowed_input() {
    for mode in [
        Mode::WrongOwner,
        Mode::Stale,
        Mode::WrongShape,
        Mode::WrongClass,
        Mode::WrongStorage,
        Mode::Alias,
    ] {
        let (provider, handle, input_probe) = StdOwner::fixture(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let error = futures::executor::block_on(super::super::var::var_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("invalid std result must not reach variance square");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error.message().contains("invalid resident reduction"));
        assert_eq!(provider.squares.load(Ordering::SeqCst), 0);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        let outputs = provider.outputs.lock().unwrap();
        if matches!(mode, Mode::Alias) {
            assert!(outputs.is_empty());
        } else {
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].retirements(), 1);
        }
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input_probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rejected_std_retirement_failure_retains_contract_and_cleanup_cause() {
    let (provider, handle, input_probe) = StdOwner::fixture(Mode::WrongShapeCleanupFailure);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let error = futures::executor::block_on(std_gpu(handle.clone(), &all_axes()))
        .expect_err("wrong shape is terminal even if cleanup fails");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().contains("invalid resident reduction"));
    assert!(error
        .message()
        .contains("provider result retirement failed"));
    assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input_probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn f32_nd_and_empty_std_results_preserve_reduced_shape_and_options() {
    let scenarios = [
        (
            vec![2, 3, 4],
            vec![1, 1, 1],
            StdAxes::All,
            StdNormalization::Population,
            ReductionNaN::Omit,
            None,
            ProviderStdNormalization::Population,
            ProviderNanMode::Omit,
        ),
        (
            vec![0, 3],
            vec![1, 3],
            StdAxes::Default,
            StdNormalization::Sample,
            ReductionNaN::Include,
            Some(0),
            ProviderStdNormalization::Sample,
            ProviderNanMode::Include,
        ),
    ];
    for (
        input_shape,
        output_shape,
        axes,
        normalization,
        nan_mode,
        expected_dim,
        expected_normalization,
        expected_nan_mode,
    ) in scenarios
    {
        let (provider, handle, input_probe) = StdOwner::fixture_with(
            Mode::Success,
            NumericElementType::F32,
            input_shape,
            output_shape.clone(),
        );
        let _selected = ThreadProviderGuard::set(Some(provider));
        let args = ParsedArguments {
            axes,
            normalization,
            nan_mode,
        };
        let result = futures::executor::block_on(std_gpu(handle.clone(), &args))
            .expect("supported resident std shape");
        let Value::GpuTensor(output) = result else {
            panic!("successful resident std stays on GPU");
        };
        assert_eq!(output.shape(), output_shape);
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(
            provider.options.lock().unwrap().as_slice(),
            &[(expected_dim, expected_normalization, expected_nan_mode)]
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input_probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn f32_nd_and_empty_var_results_stay_resident_after_valid_std_and_square() {
    for (input_shape, output_shape) in [(vec![2, 3, 4], vec![1, 3, 4]), (vec![0, 3], vec![1, 3])] {
        let (provider, handle, input_probe) = StdOwner::fixture_with(
            Mode::SquareSuccess,
            NumericElementType::F32,
            input_shape,
            output_shape.clone(),
        );
        let _selected = ThreadProviderGuard::set(Some(provider));
        let result = futures::executor::block_on(super::super::var::var_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("supported resident var shape");
        let Value::GpuTensor(output) = result else {
            panic!("successful resident var stays on GPU");
        };
        assert_eq!(output.shape(), output_shape);
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(
            provider.options.lock().unwrap().as_slice(),
            &[(
                Some(0),
                ProviderStdNormalization::Sample,
                ProviderNanMode::Include
            )]
        );
        assert_eq!(provider.squares.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input_probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn malformed_variance_square_is_terminal_and_retires_results() {
    for mode in [Mode::SquareAlias, Mode::SquareWrongShape] {
        let (provider, handle, input_probe) = StdOwner::fixture(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let error = futures::executor::block_on(super::super::var::var_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("malformed square is terminal");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error
            .message()
            .contains("square returned an invalid resident result"));
        assert_eq!(provider.squares.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        let outputs = provider.outputs.lock().unwrap();
        assert_eq!(outputs[0].retirements(), 1);
        if matches!(mode, Mode::SquareWrongShape) {
            assert_eq!(outputs.len(), 2);
            assert_eq!(outputs[1].retirements(), 1);
        } else {
            assert_eq!(outputs.len(), 1);
        }
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input_probe.retirements(), 0);
    }
}
