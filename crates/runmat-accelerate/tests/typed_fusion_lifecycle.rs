//! Actual async fusion boundary; the recorder is not a numerical GPU oracle.
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
use fixture::{cast, plan};
use runmat_accelerate::fusion::FusionStoreMaterialization;
use runmat_accelerate::fusion_exec::{execute_elementwise, FusionExecutionRequest};
use runmat_accelerate::graph::{VarBinding, VarKind};
use runmat_accelerate_api::*;
use runmat_value::Value;
use std::future::Future;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Mutex,
};
use std::task::{Context, Poll};

struct Recorder {
    next: AtomicU64,
    casts: AtomicUsize,
    mode: AtomicUsize,
    freed: Mutex<Vec<u64>>,
}
static OWNER: Recorder = Recorder {
    next: AtomicU64::new(10),
    casts: AtomicUsize::new(0),
    mode: AtomicUsize::new(0),
    freed: Mutex::new(Vec::new()),
};
static SERIAL: Mutex<()> = Mutex::new(());
impl Recorder {
    fn make(&self, id: u64, element: NumericElementType) -> GpuTensorHandle {
        GpuTensorHandle::new(vec![1, 2], self.device_id(), id)
            .with_numeric_descriptor(element, GpuTensorStorage::Real)
    }
    fn fresh(&self, element: NumericElementType) -> GpuTensorHandle {
        self.make(self.next.fetch_add(1, Ordering::SeqCst), element)
    }
}
impl AccelProvider for Recorder {
    fn device_id(&self) -> u32 {
        u32::MAX - 284
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn device_info(&self) -> String {
        "typed-fusion-lifecycle".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.freed.lock().unwrap().push(handle.buffer_id);
        Ok(())
    }
    fn query_feasibility(&self, _: &ProviderFeasibilityQuery) -> ProviderFeasibility {
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: None,
            output_bytes: None,
            dispatches: Some(1),
        })
    }
    fn fused_elementwise_multi(
        &self,
        _: &str,
        inputs: &[GpuTensorHandle],
        _: &[usize],
        _: usize,
        count: usize,
    ) -> anyhow::Result<Vec<GpuTensorHandle>> {
        set_handle_logical(&inputs[0], true);
        match self.mode.load(Ordering::SeqCst) {
            3 => return Err(unsupported_provider_operation("multi unavailable")),
            4 => anyhow::bail!("multi execution failed"),
            _ => {}
        }
        Ok((0..count)
            .map(|_| self.fresh(NumericElementType::F64))
            .collect())
    }
    fn fused_elementwise(
        &self,
        _: &str,
        _: &[GpuTensorHandle],
        _: &[usize],
        _: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        Ok(self.fresh(NumericElementType::F64))
    }
    fn unary_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let ordinal = self.casts.fetch_add(1, Ordering::SeqCst);
            if ordinal == 1 {
                clear_handle_metadata(input);
                match self.mode.load(Ordering::SeqCst) {
                    0 => futures::future::pending::<()>().await,
                    1 => anyhow::bail!("second output adapter failed"),
                    _ => {}
                }
            }
            Ok(self.fresh(NumericElementType::F32))
        })
    }
}

fn request<'a>(
    plan: &'a runmat_accelerate::fusion::FusionGroupPlan,
    input: GpuTensorHandle,
) -> FusionExecutionRequest<'a> {
    FusionExecutionRequest {
        plan,
        inputs: vec![Value::GpuTensor(input)],
        placement: None,
        runtime: None,
    }
}

#[test]
fn second_adapter_suspend_or_error_releases_every_fresh_allocation_without_publishing() {
    let _serial = SERIAL.lock().unwrap();
    let _provider = ThreadProviderGuard::set(Some(&OWNER));
    for mode in [0, 1] {
        OWNER.mode.store(mode, Ordering::SeqCst);
        OWNER.casts.store(0, Ordering::SeqCst);
        OWNER.next.store(10, Ordering::SeqCst);
        OWNER.freed.lock().unwrap().clear();
        let input = OWNER.make(1, NumericElementType::F64);
        let before = snapshot_handle_metadata(&input);
        let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        });
        let mut future = Box::pin(execute_elementwise(request(&plan, input.clone())));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        match future.as_mut().poll(&mut context) {
            Poll::Pending => assert_eq!(mode, 0),
            Poll::Ready(result) => {
                assert_eq!(mode, 1);
                assert!(result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("second output adapter failed"));
            }
        }
        drop(future);
        assert_eq!(snapshot_handle_metadata(&input), before);
        let mut freed = OWNER.freed.lock().unwrap().clone();
        freed.sort_unstable();
        assert_eq!(freed, vec![10, 11, 12]);
        for id in [10, 11, 12] {
            assert!(!runmat_accelerate::fusion_residency::is_resident(
                &OWNER.make(id, NumericElementType::F64)
            ));
        }
    }
}

#[test]
fn duplicate_materialized_bindings_adopt_one_completed_result() {
    let _serial = SERIAL.lock().unwrap();
    let _provider = ThreadProviderGuard::set(Some(&OWNER));
    OWNER.mode.store(2, Ordering::SeqCst);
    OWNER.casts.store(0, Ordering::SeqCst);
    OWNER.next.store(10, Ordering::SeqCst);
    OWNER.freed.lock().unwrap().clear();
    let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
    for index in [0, 1] {
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index,
            },
        });
    }
    let result = futures::executor::block_on(execute_elementwise(request(
        &plan,
        OWNER.make(1, NumericElementType::F64),
    )))
    .unwrap();
    let Value::GpuTensor(first) = &result.materialized_stores[0].1 else {
        panic!("resident output")
    };
    let Value::GpuTensor(second) = &result.materialized_stores[1].1 else {
        panic!("resident output")
    };
    assert_eq!(first.buffer_id, second.buffer_id);
    assert_eq!(first.descriptor.element_type, Some(NumericElementType::F32));
    let mut freed = OWNER.freed.lock().unwrap().clone();
    freed.sort_unstable();
    assert_eq!(freed, vec![10, 11]);
    let Value::GpuTensor(final_output) = result.final_value else {
        panic!("resident output")
    };
    OWNER.free(first).unwrap();
    OWNER.free(&final_output).unwrap();
}

#[test]
fn only_typed_unsupported_multi_falls_back_to_single_dispatches() {
    let _serial = SERIAL.lock().unwrap();
    let _provider = ThreadProviderGuard::set(Some(&OWNER));
    for mode in [3, 4] {
        OWNER.mode.store(mode, Ordering::SeqCst);
        OWNER.casts.store(0, Ordering::SeqCst);
        OWNER.next.store(10, Ordering::SeqCst);
        OWNER.freed.lock().unwrap().clear();
        let input = OWNER.make(1, NumericElementType::F64);
        let before = snapshot_handle_metadata(&input);
        let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
        plan.materialized_stores.push(FusionStoreMaterialization {
            value_id: 2,
            binding: VarBinding {
                kind: VarKind::Global,
                index: 0,
            },
        });
        let result =
            futures::executor::block_on(execute_elementwise(request(&plan, input.clone())));
        assert_eq!(snapshot_handle_metadata(&input), before);
        if mode == 4 {
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("multi execution failed"));
            assert_eq!(OWNER.next.load(Ordering::SeqCst), 10);
            assert_eq!(OWNER.casts.load(Ordering::SeqCst), 0);
        } else {
            let result = result.unwrap();
            assert_eq!(OWNER.next.load(Ordering::SeqCst), 14);
            let Value::GpuTensor(final_output) = result.final_value else {
                panic!("resident")
            };
            let Value::GpuTensor(stored) = &result.materialized_stores[0].1 else {
                panic!("resident")
            };
            OWNER.free(&final_output).unwrap();
            OWNER.free(stored).unwrap();
        }
    }
}

struct DefaultMulti;
impl AccelProvider for DefaultMulti {
    fn device_info(&self) -> String {
        "default-multi".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!()
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!()
    }
    fn free(&self, _: &GpuTensorHandle) -> anyhow::Result<()> {
        unreachable!()
    }
}

#[test]
fn provider_default_multi_declares_unsupported_not_execution_failure() {
    let error = DefaultMulti
        .fused_elementwise_multi("", &[], &[1, 2], 2, 2)
        .unwrap_err();
    assert!(is_unsupported_provider_operation(&error));
}
