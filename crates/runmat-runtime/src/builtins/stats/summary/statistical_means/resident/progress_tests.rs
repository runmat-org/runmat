use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    *,
};
use runmat_value::Value;
use std::error::Error;
use std::future::Future;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::task::{Context, Poll};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hook {
    MapNan,
    Min,
    Abs,
    Mean,
    ZeroConstant,
    NanConstant,
    OneConstant,
}

#[derive(Clone, Copy)]
enum HookOutcome {
    Unsupported,
    Pending,
    Alias,
}

#[derive(Clone, Copy)]
enum Invocation {
    Geometric,
    RootMeanSquaredError,
}

impl Invocation {
    fn public_name(self) -> &'static str {
        match self {
            Self::Geometric => "geomean",
            Self::RootMeanSquaredError => "rmse",
        }
    }

    fn arguments(self, left: &GpuTensorHandle, right: &GpuTensorHandle) -> Vec<Value> {
        match self {
            Self::Geometric => vec![Value::GpuTensor(left.clone())],
            Self::RootMeanSquaredError => {
                vec![
                    Value::GpuTensor(left.clone()),
                    Value::GpuTensor(right.clone()),
                ]
            }
        }
    }
}

struct Provider {
    owner: TestAllocationOwner,
    fail: Hook,
    fail_on: usize,
    outcome: HookOutcome,
    calls: AtomicUsize,
    downloads: AtomicUsize,
    probes: Mutex<Vec<(u64, RetirementProbe)>>,
    last_zero: Mutex<Option<runmat_accelerate_api::allocation::WeakGpuTensorHandle>>,
}

impl Provider {
    fn new(fail: Hook, fail_on: usize) -> &'static Self {
        Self::with_outcome(fail, fail_on, HookOutcome::Unsupported)
    }

    fn with_outcome(fail: Hook, fail_on: usize, outcome: HookOutcome) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            fail,
            fail_on,
            outcome,
            calls: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            probes: Mutex::new(Vec::new()),
            last_zero: Mutex::new(None),
        }))
    }

    fn pending_second_abs() -> &'static Self {
        Self::with_outcome(Hook::Abs, 2, HookOutcome::Pending)
    }

    fn intercept(&self, hook: Hook) -> anyhow::Result<()> {
        let ordinal = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail == hook && self.fail_on == ordinal {
            return match self.outcome {
                HookOutcome::Unsupported => {
                    Err(unsupported_provider_operation("resident stage fixture"))
                }
                HookOutcome::Pending => panic!("only the async abs hook may remain pending"),
                HookOutcome::Alias => panic!("only constant hooks inject malformed aliases"),
            };
        }
        Ok(())
    }

    fn real(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let allocated = self.owner.allocate_zeroed(
            shape.to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.probes
            .lock()
            .unwrap()
            .push((allocated.handle.buffer_id(), allocated.probe));
        Ok(allocated.handle)
    }

    fn constant(
        &self,
        prototype: &GpuTensorHandle,
        shape: &[usize],
        hook: Hook,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        let ordinal = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail == hook && self.fail_on == ordinal {
            assert!(matches!(self.outcome, HookOutcome::Alias));
            // Zero aliases its prototype; NaN/one deliberately alias the
            // earlier zero sibling, not merely their immediate prototype.
            return if hook == Hook::ZeroConstant {
                Ok(Some(prototype.clone()))
            } else {
                self.last_zero
                    .lock()
                    .unwrap()
                    .as_ref()
                    .expect("earlier zero")
                    .acquire()
                    .map(Some)
                    .map_err(Into::into)
            };
        }
        let output = self
            .real(shape)
            .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())?;
        if hook == Hook::ZeroConstant {
            *self.last_zero.lock().unwrap() = Some(output.downgrade());
        }
        Ok(Some(output))
    }

    fn logical(&self, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let allocated = self
            .owner
            .allocate_logical_zeroed(shape.to_vec(), ProviderPrecision::F64)?;
        self.probes
            .lock()
            .unwrap()
            .push((allocated.handle.buffer_id(), allocated.probe));
        Ok(allocated.handle)
    }

    fn source(&self, values: Vec<f64>, shape: Vec<usize>) -> GpuTensorHandle {
        let allocated = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(values),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        self.probes
            .lock()
            .unwrap()
            .push((allocated.handle.buffer_id(), allocated.probe));
        allocated.handle
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "resident progress fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        Err(anyhow::anyhow!("resident progress fixture does not upload"))
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            let owned = self.download_numeric(handle).await?;
            let HostNumericDataOwned::F64(data) = owned.data else {
                anyhow::bail!("resident progress fixture requires native double storage")
            };
            Ok(HostTensorOwned {
                data,
                shape: owned.shape,
                storage: owned.storage,
            })
        })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let probes = self.probes.lock().unwrap();
            let probe = &probes
                .iter()
                .find(|(id, _)| *id == handle.buffer_id())
                .ok_or_else(|| anyhow::anyhow!("unknown allocation"))?
                .1;
            let values = probe.with_data(handle, |data| match data {
                HostNumericDataView::F64(values) => values.to_vec(),
                _ => panic!("fixture is native double"),
            })?;
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(values),
                shape: handle.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }

    fn map_nan_to_zero(
        &self,
        input: &GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.intercept(Hook::MapNan)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        self.real(input.shape()).map_err(Into::into)
    }

    fn zeros_like(
        &self,
        input: &GpuTensorHandle,
        shape: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.constant(input, shape, Hook::ZeroConstant)
    }

    fn fill_like(
        &self,
        input: &GpuTensorHandle,
        value: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        assert!(value.is_nan(), "this fixture only prepares NaN constants");
        self.constant(input, input.shape(), Hook::NanConstant)
    }

    fn ones_like(
        &self,
        input: &GpuTensorHandle,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.constant(input, input.shape(), Hook::OneConstant)
    }

    fn logical_isfinite(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.logical(input.shape())
    }

    fn elem_ne<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.logical(input.shape()).map_err(Into::into)
        })
    }

    fn logical_and(
        &self,
        input: &GpuTensorHandle,
        _: &GpuTensorHandle,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.logical(input.shape())
    }

    fn unary_exp<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.real(input.shape())
        })
    }

    fn elem_ge<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.logical(input.shape()).map_err(Into::into)
        })
    }

    fn unary_log<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.real(input.shape())
        })
    }

    fn reduce_mean_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> = async move {
                self.intercept(Hook::Mean)?;
                let mut shape = input.shape().to_vec();
                shape[dim] = 1;
                self.real(&shape)
            }
            .await;
            result.map(Some).map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }

    fn reduce_min_dim<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dim: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move {
            self.intercept(Hook::Min)?;
            let mut shape = input.shape().to_vec();
            shape[dim] = 1;
            let values = self.real(&shape).map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?;
            let indices = self
                .owner
                .allocate_zeroed(shape, NumericElementType::U64, GpuTensorStorage::Real)
                .map_err(|error| {
                    runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
                })?;
            self.probes
                .lock()
                .unwrap()
                .push((indices.handle.buffer_id(), indices.probe));
            Ok(ReduceDimResult {
                values,
                indices: indices.handle,
            })
        })
    }

    fn unary_abs<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let ordinal = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail == Hook::Abs && self.fail_on == ordinal {
                match self.outcome {
                    HookOutcome::Unsupported => {
                        return Err(unsupported_provider_operation("resident stage fixture"));
                    }
                    HookOutcome::Pending => std::future::pending::<()>().await,
                    HookOutcome::Alias => panic!("only constant hooks inject malformed aliases"),
                }
            }
            self.real(input.shape())
        })
    }
}

fn assert_original_unsupported(error: &crate::RuntimeError) {
    let mut source: &(dyn Error + 'static) = error;
    loop {
        if source.is::<UnsupportedProviderOperation>() {
            return;
        }
        source = source
            .source()
            .expect("original typed unsupported cause retained");
    }
}

fn assert_constant_alias_rejected(
    provider: &Provider,
    borrowed: &[&GpuTensorHandle],
    result: Result<GpuTensorHandle, super::Failure>,
    ordinal: usize,
) {
    let super::Failure::Execution(error) =
        result.expect_err("constant alias is not a capability miss")
    else {
        panic!("constant alias must be terminal");
    };
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(error.message().contains("constructor aliased"));
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        ordinal,
        "no downstream hook"
    );
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    for handle in borrowed {
        handle.allocation_lease().validate_live().unwrap();
    }
    let probes = provider.probes.lock().unwrap();
    for (id, probe) in probes.iter() {
        if let Some(input) = borrowed.iter().find(|input| input.buffer_id() == *id) {
            assert_eq!(probe.retirements(), 0, "borrowed input remains live");
            probe
                .with_data(input, |data| {
                    let HostNumericDataView::F64(data) = data else {
                        panic!("native double fixture")
                    };
                    assert!(data.iter().all(|value| *value == 4.0));
                })
                .unwrap();
        } else {
            assert_eq!(probe.retirements(), 1, "owned intermediate retires once");
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn first_unsupported_geomean_stage_uses_documented_host_materialization() {
    let provider = Provider::new(Hook::MapNan, 1);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider.source(vec![4.0, 4.0], vec![2, 1]);
    let result = block_on(
        crate::builtins::stats::summary::statistical_means::geomean::geomean_builtin_source(
            Value::GpuTensor(input.clone()),
            vec![],
        ),
    )
    .unwrap();
    match result {
        Value::Num(value) => assert_eq!(value, 4.0),
        Value::Tensor(value) => assert_eq!(value.materialize_f64(), vec![4.0]),
        other => panic!("unexpected host fallback value: {other:?}"),
    }
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn later_geomean_and_second_rmse_operand_unsupported_are_terminal() {
    for (hook, ordinal, invocation) in [
        (Hook::Min, 2, Invocation::Geometric),
        (Hook::Abs, 2, Invocation::RootMeanSquaredError),
    ] {
        let provider = Provider::new(hook, ordinal);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let left = provider.source(vec![4.0, 4.0], vec![2, 1]);
        let right = provider.source(vec![4.0, 4.0], vec![2, 1]);
        let values = invocation.arguments(&left, &right);
        let error = block_on(runmat_runtime_call(invocation.public_name(), &values)).unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_original_unsupported(&error);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        left.allocation_lease().validate_live().unwrap();
        right.allocation_lease().validate_live().unwrap();
        assert!(provider
            .probes
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, _)| *id != left.buffer_id() && *id != right.buffer_id())
            .all(|(_, probe)| probe.retirements() == 1));
    }
    // The four geometric constructor sites: zero aliases its prototype, and
    // NaN aliases an earlier zero sibling. Neither may reach the next hook.
    for (hook, ordinal, singleton) in [
        (Hook::ZeroConstant, 1, true),
        (Hook::NanConstant, 3, true),
        (Hook::ZeroConstant, 3, false),
        (Hook::NanConstant, 8, false),
    ] {
        let provider = Provider::with_outcome(hook, ordinal, HookOutcome::Alias);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider.source(vec![4.0, 4.0], vec![2, 1]);
        let options = super::super::options::Options {
            axes: if singleton {
                super::super::options::Axes::Dims(vec![3])
            } else {
                super::super::options::Axes::Default
            },
            missing: super::super::options::MissingMode::Include,
            weights: None,
            typed_integer_control: false,
        };
        let result = block_on(super::numerics::run(
            super::ReductionKind::Geometric,
            provider,
            &input,
            &options,
            &mut super::ResidentStageProgress::default(),
        ));
        assert_constant_alias_rejected(provider, &[&input], result, ordinal);
        drop(input);
        assert!(provider
            .probes
            .lock()
            .unwrap()
            .iter()
            .all(|(_, probe)| probe.retirements() == 1));
    }
    // The shared scale sites: a zero prototype alias and a one/zero sibling
    // alias must be rejected for RMS/harmonic/RMSE before comparison/select.
    for (hook, ordinal) in [(Hook::ZeroConstant, 2), (Hook::OneConstant, 5)] {
        let provider = Provider::with_outcome(hook, ordinal, HookOutcome::Alias);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider.source(vec![4.0, 4.0], vec![2, 1]);
        let scale = provider.source(vec![4.0], vec![1, 1]);
        let result = block_on(super::usable_scale(
            super::ReductionKind::RootMeanSquare,
            provider,
            &input,
            scale.clone(),
            &[],
            &mut super::ResidentStageProgress::default(),
        ));
        assert_constant_alias_rejected(provider, &[&input, &scale], result, ordinal);
        drop((input, scale));
        assert!(provider
            .probes
            .lock()
            .unwrap()
            .iter()
            .all(|(_, probe)| probe.retirements() == 1));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn first_and_second_nested_mean_axes_retain_outer_effect_boundary() {
    for ordinal in [7, 8] {
        let provider = Provider::new(Hook::Mean, ordinal);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider.source(vec![4.0; 4], vec![2, 2]);
        let error = block_on(crate::call_builtin_async(
            "geomean",
            &[Value::GpuTensor(input.clone()), Value::from("all")],
        ))
        .unwrap_err();
        assert_eq!(provider.calls.load(Ordering::SeqCst), ordinal);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_original_unsupported(&error);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        input.allocation_lease().validate_live().unwrap();
        let probes = provider.probes.lock().unwrap();
        let temporaries: Vec<_> = probes
            .iter()
            .filter(|(id, _)| *id != input.buffer_id())
            .collect();
        assert!(
            !temporaries.is_empty(),
            "outer stages produced real temporaries"
        );
        assert!(temporaries
            .iter()
            .all(|(_, probe)| probe.retirements() == 1));
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_second_rmse_operand_cancellation_retires_prior_stage_without_replay() {
    let provider = Provider::pending_second_abs();
    let _selected = ThreadProviderGuard::set(Some(provider));
    let left = provider.source(vec![4.0, 4.0], vec![2, 1]);
    let right = provider.source(vec![4.0, 4.0], vec![2, 1]);
    let arguments = [
        Value::GpuTensor(left.clone()),
        Value::GpuTensor(right.clone()),
    ];
    let mut request = Box::pin(crate::call_builtin_async("rmse", &arguments));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(request.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    left.allocation_lease().validate_live().unwrap();
    right.allocation_lease().validate_live().unwrap();
    drop(request);
    let probes = provider.probes.lock().unwrap();
    let temporary: Vec<_> = probes
        .iter()
        .filter(|(id, _)| *id != left.buffer_id() && *id != right.buffer_id())
        .collect();
    assert!(
        !temporary.is_empty(),
        "first operand produced a real temporary"
    );
    assert!(temporary.iter().all(|(_, probe)| probe.retirements() == 1));
    assert!(probes
        .iter()
        .filter(|(id, _)| *id == left.buffer_id() || *id == right.buffer_id())
        .all(|(_, probe)| probe.retirements() == 0));
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}

async fn runmat_runtime_call(
    name: &str,
    values: &[Value],
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    crate::call_builtin_async(name, values).await
}
