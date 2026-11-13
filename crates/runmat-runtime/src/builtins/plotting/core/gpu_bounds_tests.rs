use super::*;
use futures::{executor::block_on, task::noop_waker, FutureExt};
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, HostTensorView, ProviderPrecision, ThreadProviderGuard,
};
use std::sync::Mutex;
use std::task::{Context, Poll};

#[derive(Clone)]
enum Mode {
    Ready,
    PendingMax,
    ErrorMax,
    Alias(GpuTensorHandle),
    PendingDownload,
    ErrorDownload,
    RejectLogicalView,
    #[cfg(feature = "wgpu")]
    ExportNone,
    #[cfg(feature = "wgpu")]
    ExportError,
    #[cfg(feature = "wgpu")]
    ExportAfterEffect,
}
struct Provider {
    inner: InProcessProvider,
    mode: Mutex<Mode>,
    produced: Mutex<Vec<runmat_accelerate_api::GpuTensorWireDescriptor>>,
}
impl AccelProvider for Provider {
    #[cfg(feature = "wgpu")]
    fn export_wgpu_buffer(
        &self,
        _: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<runmat_accelerate_api::WgpuBufferRef>>
    {
        let mode = self.mode.lock().unwrap().clone();
        match mode {
            Mode::ExportError | Mode::ExportAfterEffect => {
                let error = runmat_accelerate_api::ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "injected export failure",
                    ),
                );
                Err(if matches!(mode, Mode::ExportAfterEffect) {
                    error.after_effect()
                } else {
                    error
                })
            }
            _ => Ok(None),
        }
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, input: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(input)
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        let mode = self.mode.lock().unwrap().clone();
        Box::pin(async move {
            match mode {
                Mode::PendingDownload => futures::future::pending().await,
                Mode::ErrorDownload => anyhow::bail!("injected download failure"),
                _ => self.inner.download(input).await,
            }
        })
    }
    fn reduce_min<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        let mode = self.mode.lock().unwrap().clone();
        Box::pin(async move {
            match mode {
                Mode::Alias(handle) => Ok(handle),
                Mode::RejectLogicalView => {
                    let output = self.inner.reduce_min(input).await?;
                    assert!(input
                        .with_view(input.view().clone().with_logical(true))
                        .is_err());

                    self.produced.lock().unwrap().push(output.wire_descriptor());
                    Ok(output)
                }
                _ => {
                    let output = self.inner.reduce_min(input).await?;
                    self.produced.lock().unwrap().push(output.wire_descriptor());
                    Ok(output)
                }
            }
        })
    }
    fn reduce_max<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        let mode = self.mode.lock().unwrap().clone();
        Box::pin(async move {
            match mode {
                Mode::PendingMax => {
                    assert!(input
                        .with_view(input.view().clone().with_logical(true))
                        .is_err());
                    futures::future::pending().await
                }
                Mode::ErrorMax => Err(anyhow::anyhow!("injected max failure").into()),
                Mode::RejectLogicalView => {
                    assert!(!runmat_accelerate_api::handle_is_logical(input));
                    assert!(runmat_accelerate_api::handle_transpose_info(input).is_none());
                    {
                        let output = self.inner.reduce_max(input).await?;
                        self.produced.lock().unwrap().push(output.wire_descriptor());
                        Ok(output)
                    }
                }
                _ => {
                    let output = self.inner.reduce_max(input).await?;
                    self.produced.lock().unwrap().push(output.wire_descriptor());
                    Ok(output)
                }
            }
        })
    }
}
fn provider(mode: Mode) -> &'static Provider {
    Box::leak(Box::new(Provider {
        inner: InProcessProvider::new(),
        mode: Mutex::new(mode),
        produced: Mutex::new(Vec::new()),
    }))
}
fn upload(provider: &Provider) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[1, 2],
        })
        .unwrap()
}

#[test]
fn bounds_reject_logical_reinterpretation_and_keep_admitted_output_type() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider(Mode::RejectLogicalView);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = upload(provider);
    assert_eq!(
        block_on(axis_bounds_protecting(&input, &[&input], "stem")).unwrap(),
        (1.0, 2.0)
    );
    assert!(!runmat_accelerate_api::handle_is_logical(&input));
    let produced = provider.produced.lock().unwrap().clone();
    assert_eq!(produced.len(), 2);
    assert!(produced
        .iter()
        .all(|wire| provider.inner.acquire_live(wire).is_err()));
    assert!(block_on(provider.inner.download(&input)).is_ok());
    let minimum = 1.0 + 2.0f64.powi(-40);
    let maximum = 2.0 + 2.0f64.powi(-40);
    let precise = provider
        .upload(&HostTensorView {
            data: &[minimum, maximum],
            shape: &[1, 2],
        })
        .unwrap();
    provider.produced.lock().unwrap().clear();
    let bounds = block_on(super::super::numeric_bounds_protecting(
        &precise,
        &[&precise],
        "hist",
    ))
    .expect("native numerical bounds");
    assert_eq!((bounds.min, bounds.max), (minimum, maximum));
    assert_eq!(bounds.rendering(), (1.0f32, 2.0f32));
    let produced = provider.produced.lock().unwrap().clone();
    assert_eq!(
        produced.len(),
        2,
        "one reduction pair, not a second numerical scan"
    );
    assert!(produced
        .iter()
        .all(|wire| provider.inner.acquire_live(wire).is_err()));
    assert_eq!(
        block_on(provider.inner.download(&precise)).unwrap().data,
        vec![minimum, maximum]
    );
    precise.release().unwrap();
    input.release().unwrap();
}

#[test]
fn bounds_drop_error_and_cross_coordinate_alias_preserve_borrowed_inputs() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider(Mode::Ready);
    let _current = ThreadProviderGuard::set(Some(provider));
    let x = upload(provider);
    let y = upload(provider);
    #[cfg(feature = "wgpu")]
    {
        // A missing exact rendering context is pure absence even if an export
        // hook would fail operationally if called. Do not probe that hook.
        *provider.mode.lock().unwrap() = Mode::ExportError;
        assert!(super::super::GpuBufferExports::prepare(&[&x, &y], "plot")
            .unwrap()
            .is_none());
        use crate::builtins::plotting::op_common::surface_inputs::{
            AxisSource, PreparedSurfaceGpu,
        };
        let horizontal = AxisSource::Host(vec![1.0, 2.0]);
        let singleton = AxisSource::Host(vec![1.0]);
        assert!(block_on(PreparedSurfaceGpu::prepare(
            "surf",
            &horizontal,
            &singleton,
            &x
        ))
        .unwrap()
        .is_none());
        assert!(
            block_on(PreparedSurfaceGpu::prepare(
                "surf",
                &horizontal,
                &horizontal,
                &x
            ))
            .is_err(),
            "invalid grid geometry must fail before context/export/bounds work"
        );
        assert!(provider.produced.lock().unwrap().is_empty());
    }
    #[cfg(feature = "wgpu")]
    for mode in [Mode::ExportNone, Mode::ExportError, Mode::ExportAfterEffect] {
        *provider.mode.lock().unwrap() = mode.clone();
        let mut exports = super::super::GpuBufferExports::default();
        let result = exports.export(&x, "plot");
        if matches!(mode, Mode::ExportNone) {
            assert!(result.unwrap().is_none());
        } else {
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("operational export failure was swallowed"),
            };
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error.to_string().contains("injected export failure"));
        }
        for completed in [false, true] {
            let mut stage = super::super::GpuBufferExports::default();
            if completed {
                stage.record_completed_effect();
            }
            let source = runmat_accelerate_api::WgpuAccessError::WrongDevice;
            let error = stage
                .complete::<()>(
                    Err(crate::builtins::plotting::plotting_error_with_source(
                        "plot",
                        "original emission refusal",
                        source,
                    )),
                    "plot",
                )
                .unwrap_err();
            assert_eq!(error.message(), "original emission refusal");
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            let mut cursor: &(dyn std::error::Error + 'static) = &error;
            let mut found_source = false;
            let mut found_phase = false;
            loop {
                found_source |= cursor.downcast_ref::<runmat_accelerate_api::WgpuAccessError>()
                    == Some(&source);
                if let Some(provider) =
                    cursor.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                {
                    found_phase |= provider.is_after_effect();
                }
                let Some(next) = cursor.source() else {
                    break;
                };
                cursor = next;
            }
            assert!(found_source, "typed access cause must survive completion");
            assert_eq!(found_phase, completed);
        }
        assert!(provider.produced.lock().unwrap().is_empty());
        assert!(block_on(provider.inner.download(&x)).is_ok());
        assert!(block_on(provider.inner.download(&y)).is_ok());
    }
    for mode in [Mode::PendingMax, Mode::ErrorMax, Mode::Alias(y.clone())] {
        *provider.mode.lock().unwrap() = mode.clone();
        provider.produced.lock().unwrap().clear();
        let protected = [&x, &y];
        let mut future = Box::pin(axis_bounds_protecting(&x, &protected, "stem"));
        if matches!(mode, Mode::PendingMax) {
            assert!(matches!(
                future.poll_unpin(&mut Context::from_waker(&noop_waker())),
                Poll::Pending
            ));
            drop(future);
        } else {
            assert!(block_on(future).is_err());
        }
        let retired = provider.produced.lock().unwrap().clone();
        assert_eq!(retired.len(), usize::from(!matches!(mode, Mode::Alias(_))));
        assert!(retired
            .iter()
            .all(|wire| provider.inner.acquire_live(wire).is_err()));
        assert!(!runmat_accelerate_api::handle_is_logical(&x));
        assert!(block_on(provider.inner.download(&x)).is_ok());
        assert!(block_on(provider.inner.download(&y)).is_ok());
    }
    x.release().unwrap();
    y.release().unwrap();
}

#[test]
fn stem_body_preparation_can_suspend_fail_and_drop_before_plot_mutation() {
    use crate::builtins::plotting::tests::{ensure_plot_test_env, lock_plot_registry};
    use crate::builtins::plotting::{clear_figure, clone_figure, current_figure_handle};
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let _plots = lock_plot_registry();
    ensure_plot_test_env();
    let _ = clear_figure(None);
    let figure = current_figure_handle();
    let count = || clone_figure(figure).unwrap().plots().count();
    let initial = count();
    let provider = provider(Mode::PendingDownload);
    let _current = ThreadProviderGuard::set(Some(provider));
    let input = upload(provider);
    // Mixed host/resident coordinates choose the owner-gather route without
    // depending on whether the test machine has a WGPU rendering context.
    let args = [
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::GpuTensor(input.clone()),
    ];
    let mut future = Box::pin(crate::builtins::plotting::stem::stem_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args.to_vec()),
    ));
    assert!(matches!(
        future.poll_unpin(&mut Context::from_waker(&noop_waker())),
        Poll::Pending
    ));
    assert_eq!(count(), initial);
    drop(future);
    assert_eq!(count(), initial);
    assert!(provider.produced.lock().unwrap().is_empty());
    *provider.mode.lock().unwrap() = Mode::ErrorDownload;
    assert!(block_on(crate::builtins::plotting::stem::stem_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args.to_vec())
    ))
    .is_err());
    assert_eq!(count(), initial);
    *provider.mode.lock().unwrap() = Mode::Ready;
    block_on(crate::builtins::plotting::stem::stem_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args.to_vec()),
    ))
    .unwrap();
    assert_eq!(count(), initial + 1);
    assert!(block_on(provider.inner.download(&input)).is_ok());
    assert!(provider.produced.lock().unwrap().is_empty());
    input.release().unwrap();
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires an actual WGPU device"]
fn plot_input_clones_retain_exact_export_allocation_until_last_drop() {
    use runmat_accelerate_api::{GpuTensorLayout, TransposeInfo};
    use runmat_plot::gpu::{line::LineGpuInputs, GpuInputBuffer, ScalarType};
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("actual WGPU qualification requires a device");
    for transpose in [false, true] {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: &[2, 3],
            })
            .unwrap();
        let input = if transpose {
            input
                .with_view(input.view().clone().with_shape(vec![3, 2]).with_layout(
                    GpuTensorLayout::Transpose(TransposeInfo {
                        base_rows: 2,
                        base_cols: 3,
                    }),
                ))
                .unwrap()
        } else {
            input
        };
        let (_, mut exports) = super::super::GpuBufferExports::prepare(&[&input], "plot")
            .unwrap()
            .expect("actual owner/context supports floating rendering");
        let exported = exports.export(&input, "plot").unwrap().unwrap();
        if transpose {
            use runmat_accelerate_api::{
                GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
            };
            let integer = provider
                .upload_numeric(&HostNumericTensorView {
                    data: HostNumericDataView::U64(&[9_007_199_254_740_993]),
                    shape: &[1, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            assert!(
                super::super::GpuBufferExports::prepare(&[&input, &integer], "plot")
                    .unwrap()
                    .is_none(),
                "known integer carrier declines before any view export"
            );
            let protected = exports.protected_handles();
            assert!(protected
                .iter()
                .any(|handle| handle.shares_allocation_with(&input)));
            assert!(protected
                .iter()
                .any(|handle| handle.shares_allocation_with(exported.owning_handle())));
            drop(protected);
            let materialized = exports.export(&input, "plot").unwrap().unwrap();
            assert!(!materialized.owning_handle().shares_allocation_with(&input));
            let repeated = exports.export(&input, "plot").unwrap().unwrap();
            assert!(
                materialized
                    .owning_handle()
                    .shares_allocation_with(repeated.owning_handle()),
                "nested consumers retain one actual materialization"
            );
            let error = match exports.export(&integer, "plot") {
                Err(error) => error,
                Ok(_) => panic!("absence after materialization authorized host replay"),
            };
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            input.allocation_lease().validate_live().unwrap();
            integer.allocation_lease().validate_live().unwrap();
            drop((materialized, repeated, integer));
        }
        assert_eq!(
            exported.owning_handle().shares_allocation_with(&input),
            !transpose
        );
        let wire = exported.owning_handle().wire_descriptor();
        let scalar = ScalarType::from_is_f64(exported.precision == ProviderPrecision::F64);
        let buffer = GpuInputBuffer::from(exported);
        let retained = LineGpuInputs {
            x_buffer: buffer.clone(),
            y_buffer: buffer.clone(),
            len: 6,
            scalar,
        };
        drop(buffer);
        drop(exports);
        drop(input);
        let live = provider.acquire_live(&wire).unwrap();
        let expected = if transpose {
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
        } else {
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        };
        assert_eq!(block_on(provider.download(&live)).unwrap().data, expected);
        drop(live);
        let cached = retained.clone();
        drop(retained);
        assert!(provider.acquire_live(&wire).is_ok());
        drop(cached);
        assert!(provider.acquire_live(&wire).is_err());
    }
}
