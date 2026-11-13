use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{HostNumericDataOwned, HostNumericDataView, HostNumericTensorView};
use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};

struct AliasingCastProvider {
    calls: AtomicUsize,
}

impl AccelProvider for AliasingCastProvider {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "aliasing resident cast test fixture".into()
    }

    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("test fixture cannot upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("test fixture cannot download") })
    }

    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        ProviderFeasibility::Supported {
            estimate: ProviderResourceEstimate {
                transient_bytes: Some(16),
                output_bytes: Some(16),
                dispatches: Some(1),
            },
        }
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(input.clone()) })
    }
}

static ALIASING_CAST_PROVIDER: AliasingCastProvider = AliasingCastProvider {
    calls: AtomicUsize::new(0),
};

struct ChangingCastProvider {
    queries: AtomicUsize,
    conversions: AtomicUsize,
}

impl AccelProvider for ChangingCastProvider {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "changing resident cast test fixture".into()
    }

    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("test fixture cannot upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("test fixture cannot download") })
    }

    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        if self.queries.fetch_add(1, Ordering::SeqCst) == 0 {
            ProviderFeasibility::Supported {
                estimate: ProviderResourceEstimate {
                    transient_bytes: Some(16),
                    output_bytes: Some(16),
                    dispatches: Some(1),
                },
            }
        } else {
            ProviderFeasibility::Rejected {
                rejection: runmat_accelerate_api::ProviderRejection {
                    code: runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
                    detail: None,
                },
            }
        }
    }

    fn unary_double<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.conversions.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(runmat_accelerate_api::ProviderExecutionError::from(
                anyhow::anyhow!("cast must not run after feasibility changes"),
            ))
        })
    }
}

static CHANGING_CAST_PROVIDER: ChangingCastProvider = ChangingCastProvider {
    queries: AtomicUsize::new(0),
    conversions: AtomicUsize::new(0),
};

struct WrongClassCastProvider {
    owner: OnceLock<runmat_accelerate_api::test_support::TestAllocationOwner>,
}

thread_local! {
    static WRONG_CLASS_RESULT_PROBE: RefCell<Option<runmat_accelerate_api::test_support::RetirementProbe>> = const { RefCell::new(None) };
}

impl WrongClassCastProvider {
    fn owner(&self) -> &runmat_accelerate_api::test_support::TestAllocationOwner {
        self.owner
            .get_or_init(runmat_accelerate_api::test_support::TestAllocationOwner::new)
    }
}

impl AccelProvider for WrongClassCastProvider {
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "wrong-class resident cast test fixture".into()
    }

    fn device_id(&self) -> u32 {
        self.owner().device_id()
    }

    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("test fixture cannot upload")
    }

    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("test fixture cannot download") })
    }

    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        ProviderFeasibility::Supported {
            estimate: ProviderResourceEstimate {
                transient_bytes: Some(16),
                output_bytes: Some(16),
                dispatches: Some(1),
            },
        }
    }

    fn unary_double<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let result = self
                .owner()
                .allocate_zeroed(
                    input.shape().to_vec(),
                    runmat_accelerate_api::NumericElementType::F32,
                    GpuTensorStorage::Real,
                )
                .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
            WRONG_CLASS_RESULT_PROBE.with(|probe| *probe.borrow_mut() = Some(result.probe));
            Ok(result.handle)
        })
    }
}

static WRONG_CLASS_CAST_PROVIDER: WrongClassCastProvider = WrongClassCastProvider {
    owner: OnceLock::new(),
};

fn requirement(
    source: ProviderPrecision,
    target: ProviderPrecision,
    transient_bytes: u64,
    output_bytes: u64,
    dispatches: u32,
) -> CastRequirement {
    CastRequirement {
        source,
        target,
        transient_bytes,
        output_bytes,
        dispatches,
    }
}

#[test]
fn cast_only_bound_is_checked_and_atomic() {
    let first = requirement(ProviderPrecision::F32, ProviderPrecision::F64, 4, 8, 1);
    let mut bound = CastBudget::default();
    bound.include(first, "dlgradient").unwrap();
    let retained = bound;
    assert!(bound
        .include(
            requirement(
                ProviderPrecision::F32,
                ProviderPrecision::F64,
                1,
                u64::MAX,
                1
            ),
            "dlgradient",
        )
        .is_err());
    assert_eq!(bound.transient_bytes, retained.transient_bytes);
    assert_eq!(bound.output_bytes, retained.output_bytes);
    assert_eq!(bound.dispatches, retained.dispatches);
    assert!(bound
        .consume(
            requirement(ProviderPrecision::F64, ProviderPrecision::F32, 4, 8, 1),
            first,
            "dlgradient",
        )
        .is_err());
    assert_eq!(bound.transient_bytes, retained.transient_bytes);
    assert_eq!(bound.output_bytes, retained.output_bytes);
    assert_eq!(bound.dispatches, retained.dispatches);
}

#[test]
fn resident_widening_uses_admitted_native_provider_cast_without_gather_or_alias() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1.25, -2.5]),
                shape: &[2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let requirement = admit(
            provider,
            ProviderPrecision::F32,
            ProviderPrecision::F64,
            &[2, 1],
            "dlgradient",
        )
        .unwrap()
        .expect("widening requires a physical cast");
        let mut budget = CastBudget::default();
        budget.include(requirement, "dlgradient").unwrap();
        let before = provider.telemetry_snapshot();
        let converted = block_on(execute(
            provider,
            input.clone(),
            ProviderPrecision::F64,
            std::slice::from_ref(&input),
            requirement,
            &mut budget,
            "dlgradient",
        ))
        .unwrap();
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            before.download_bytes
        );
        assert_eq!(converted.shape(), &[2, 1]);
        assert_eq!(handle_precision(&converted), Some(ProviderPrecision::F64));
        assert!(!converted.shares_allocation_with(&input));
        assert_eq!(budget.output_bytes, 0);
        assert_eq!(budget.transient_bytes, 0);
        assert_eq!(budget.dispatches, 0);
        assert_eq!(
            block_on(provider.download_numeric(&converted))
                .unwrap()
                .data,
            HostNumericDataOwned::F64(vec![1.25, -2.5])
        );
        assert_eq!(
            block_on(provider.download_numeric(&input)).unwrap().data,
            HostNumericDataOwned::F32(vec![1.25, -2.5])
        );
    });
}

#[test]
fn resident_widening_rejects_f32_only_owner_before_provider_effects() {
    crate::builtins::common::test_support::with_f32_test_provider(|provider| {
        let before = provider.telemetry_snapshot();
        assert!(admit(
            provider,
            ProviderPrecision::F32,
            ProviderPrecision::F64,
            &[2, 1],
            "dlgradient",
        )
        .is_err());
        assert_eq!(provider.telemetry_snapshot(), before);
    });
}

#[test]
fn changed_cast_source_is_rejected_before_provider_work() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[2.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let admitted = requirement(ProviderPrecision::F64, ProviderPrecision::F32, 128, 8, 1);
        let mut budget = CastBudget::default();
        budget.include(admitted, "dlgradient").unwrap();
        let before = provider.telemetry_snapshot();
        assert!(block_on(execute(
            provider,
            input.clone(),
            ProviderPrecision::F64,
            std::slice::from_ref(&input),
            admitted,
            &mut budget,
            "dlgradient",
        ))
        .is_err());
        assert_eq!(provider.telemetry_snapshot(), before);
        assert_eq!(budget.output_bytes, 8);
    });
}

#[test]
fn cast_requirement_smaller_than_current_provider_demand_rejects_before_work() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[2.0, 3.0]),
                shape: &[2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let mut admitted = admit(
            provider,
            ProviderPrecision::F32,
            ProviderPrecision::F64,
            input.shape(),
            "dlgradient",
        )
        .unwrap()
        .unwrap();
        admitted.output_bytes -= 1;
        let mut budget = CastBudget::default();
        budget.include(admitted, "dlgradient").unwrap();
        let before = provider.telemetry_snapshot();
        assert!(block_on(execute(
            provider,
            input.clone(),
            ProviderPrecision::F64,
            std::slice::from_ref(&input),
            admitted,
            &mut budget,
            "dlgradient",
        ))
        .is_err());
        assert_eq!(provider.telemetry_snapshot(), before);
        assert_eq!(budget.output_bytes, admitted.output_bytes);
    });
}

#[test]
fn provider_feasibility_change_rejects_before_cast_and_retains_input() {
    use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};

    CHANGING_CAST_PROVIDER.queries.store(0, Ordering::SeqCst);
    CHANGING_CAST_PROVIDER
        .conversions
        .store(0, Ordering::SeqCst);
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let admitted = admit(
        &CHANGING_CAST_PROVIDER,
        ProviderPrecision::F32,
        ProviderPrecision::F64,
        input.handle.shape(),
        "dlgradient",
    )
    .unwrap()
    .unwrap();
    let mut budget = CastBudget::default();
    budget.include(admitted, "dlgradient").unwrap();
    assert!(block_on(execute(
        &CHANGING_CAST_PROVIDER,
        input.handle.clone(),
        ProviderPrecision::F64,
        std::slice::from_ref(&input.handle),
        admitted,
        &mut budget,
        "dlgradient",
    ))
    .is_err());
    assert_eq!(CHANGING_CAST_PROVIDER.queries.load(Ordering::SeqCst), 2);
    assert_eq!(CHANGING_CAST_PROVIDER.conversions.load(Ordering::SeqCst), 0);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn wrong_class_cast_result_is_retired_without_releasing_borrowed_input() {
    use runmat_accelerate_api::NumericElementType;

    let provider = &WRONG_CLASS_CAST_PROVIDER;
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner()
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let admitted = admit(
        provider,
        ProviderPrecision::F32,
        ProviderPrecision::F64,
        input.handle.shape(),
        "dlgradient",
    )
    .unwrap()
    .unwrap();
    let mut budget = CastBudget::default();
    budget.include(admitted, "dlgradient").unwrap();
    assert!(block_on(execute(
        provider,
        input.handle.clone(),
        ProviderPrecision::F64,
        std::slice::from_ref(&input.handle),
        admitted,
        &mut budget,
        "dlgradient",
    ))
    .is_err());
    let probe = WRONG_CLASS_RESULT_PROBE.with(|probe| probe.borrow_mut().take().unwrap());
    assert_eq!(probe.retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn malformed_cast_alias_retires_only_result_reference_and_preserves_borrowed_input() {
    use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};

    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let admitted = requirement(ProviderPrecision::F32, ProviderPrecision::F64, 16, 16, 1);
    let mut budget = CastBudget::default();
    budget.include(admitted, "dlgradient").unwrap();
    let before = ALIASING_CAST_PROVIDER.calls.load(Ordering::SeqCst);
    assert!(block_on(execute(
        &ALIASING_CAST_PROVIDER,
        allocation.handle.clone(),
        ProviderPrecision::F64,
        std::slice::from_ref(&allocation.handle),
        admitted,
        &mut budget,
        "dlgradient",
    ))
    .is_err());
    assert_eq!(
        ALIASING_CAST_PROVIDER.calls.load(Ordering::SeqCst),
        before + 1
    );
    assert_eq!(allocation.probe.retirements(), 0);
    allocation
        .handle
        .allocation_lease()
        .validate_live()
        .unwrap();
}
