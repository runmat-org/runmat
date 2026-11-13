//! Provider-agnostic owner, lease, observation, and cancellation witnesses.
use super::*;
use crate::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, HostTensorView,
};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Mutex,
};

struct Fixture {
    owner: TestAllocationOwner,
    mode: AtomicU8,
    produced: Mutex<Vec<RetirementProbe>>,
    alias_target: Mutex<Option<GpuTensorHandle>>,
}

impl AccelProvider for Fixture {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "real binary API fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no staging")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("no readback")
    }
    fn propose_real_binary(
        &self,
        request: &ProviderRealBinaryRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealBinaryPhysicalProposal>> {
        if self.mode.load(Ordering::Relaxed) == 9 {
            return Ok(None);
        }
        let shape = request.lhs.representation.shape.clone();
        let output = ProviderRepresentation {
            element_type: request.lhs.representation.element_type,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            shape,
        };
        Ok(Some(ProviderRealBinaryPhysicalProposal {
            route: ProviderRealBinaryRoute::SameShape,
            output: output.clone(),
            output_buffer_bytes: output.checked_byte_len().unwrap(),
            workgroup_size: 64,
            chunk_capacity: 64,
            uniform_allocations: 1,
            uniform_buffer_bytes: 256,
            dispatches: 1,
            pre_output_submissions: 2,
            queue_submissions: 3,
        }))
    }
    fn execute_real_binary_recipe<'a>(
        &'a self,
        lhs: &'a GpuTensorHandle,
        rhs: &'a GpuTensorHandle,
        recipe: &'a ProviderRealBinaryRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> crate::ProviderExecutionFuture<'a, ProviderExecutedRealBinary> {
        Box::pin(async move {
            let mode = self.mode.load(Ordering::Relaxed);
            let output = if matches!(mode, 12 | 15) {
                self.alias_target.lock().unwrap().as_ref().unwrap().clone()
            } else if mode == 2 {
                lhs.with_view(lhs.view().clone()).unwrap()
            } else if mode == 8 {
                rhs.with_view(rhs.view().clone()).unwrap()
            } else {
                let allocation = self
                    .owner
                    .allocate_zeroed(
                        if mode == 14 {
                            vec![1, 2, 1]
                        } else {
                            lhs.shape().to_vec()
                        },
                        if mode == 13 {
                            NumericElementType::F64
                        } else {
                            lhs.physical().element_type()
                        },
                        GpuTensorStorage::Real,
                    )
                    .unwrap();
                self.produced.lock().unwrap().push(allocation.probe);
                allocation.handle
            };
            if mode == 3 {
                return Err(ProviderExecutionError::from(std::io::Error::other(
                    "original binary hook cause",
                ))
                .after_effect());
            }
            if mode == 4 {
                std::future::pending::<()>().await;
            }
            let plan = recipe.physical();
            Ok(ProviderExecutedRealBinary {
                observed: ProviderRealBinaryObserved {
                    output_buffer_bytes: Some(
                        output.physical().requested_bytes() + u64::from(mode == 5),
                    ),
                    uniform_allocations: plan.uniform_allocations + u32::from(mode == 6),
                    uniform_buffer_bytes: plan.uniform_buffer_bytes + u64::from(mode == 1),
                    dispatches: plan.dispatches + u32::from(mode == 7),
                    pre_output_submissions: plan.pre_output_submissions + u32::from(mode == 10),
                    queue_submissions: plan.queue_submissions + u32::from(mode == 11),
                },
                output,
            })
        })
    }
}

fn new_fixture() -> &'static Fixture {
    let fixture = Box::leak(Box::new(Fixture {
        owner: TestAllocationOwner::new(),
        mode: AtomicU8::new(0),
        produced: Mutex::new(Vec::new()),
        alias_target: Mutex::new(None),
    }));
    // SAFETY: process-lifetime fixture with its own private device identity.
    unsafe { crate::register_device_provider(fixture) };
    fixture
}

fn input(fixture: &Fixture) -> crate::test_support::TestAllocation {
    fixture
        .owner
        .allocate_zeroed(
            vec![2, 1, 1],
            NumericElementType::F32,
            GpuTensorStorage::Real,
        )
        .unwrap()
}

fn request(operation: ProviderRealBinaryOperation) -> ProviderRealBinaryRequest {
    let operand = ProviderRealBinaryInput {
        representation: ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            shape: vec![2, 1, 1],
        },
        buffer_bytes: 8,
    };
    ProviderRealBinaryRequest {
        operation,
        lhs: operand.clone(),
        rhs: operand,
    }
}

fn ready<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(output) => output,
        std::task::Poll::Pending => panic!("deterministic fixture unexpectedly suspended"),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn operation_class_shape_extent_and_owner_are_exact() {
    let fixture = new_fixture();
    let sub = ProviderRealBinaryRecipe::plan(fixture, request(ProviderRealBinaryOperation::Sub))
        .unwrap()
        .unwrap();
    let div = ProviderRealBinaryRecipe::plan(fixture, request(ProviderRealBinaryOperation::Div))
        .unwrap()
        .unwrap();
    assert_ne!(sub.identity().unwrap(), div.identity().unwrap());
    for mutate in [0, 1, 2, 3, 4, 5] {
        let mut changed = request(ProviderRealBinaryOperation::Sub);
        match mutate {
            0 => changed.rhs.representation.element_type = ProviderElementType::F64,
            1 => changed.lhs.representation.storage = ProviderStorage::DenseComplexInterleaved,
            2 => changed.rhs.representation.layout = ProviderLayout::Strided,
            3 => changed.rhs.representation.shape = vec![0, 1, 1],
            4 => changed.lhs.buffer_bytes = 1,
            5 => changed.rhs.representation.residency = ProviderResidency::Unknown,
            _ => unreachable!(),
        }
        assert!(ProviderRealBinaryRecipe::plan(fixture, changed)
            .unwrap()
            .is_none());
    }
    let lhs = input(fixture);
    let rhs = input(fixture);
    let result = ready(sub.execute(&lhs.handle, &rhs.handle)).unwrap();
    assert_eq!(result.observed.queue_submissions, 3);
    let foreign = new_fixture();
    assert!(!sub.is_owned_by(foreign));
    let foreign_input = input(foreign);
    assert!(ready(sub.execute(&foreign_input.handle, &rhs.handle)).is_err());
    assert_eq!(foreign_input.probe.retirements(), 0);
    fixture.mode.store(9, Ordering::Relaxed);
    let error = ready(sub.execute(&lhs.handle, &rhs.handle)).err().unwrap();
    assert!(
        !error.is_after_effect(),
        "changed proposal rejects before effects"
    );
    assert_eq!(fixture.produced.lock().unwrap().len(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn observed_fields_aliases_and_typed_hook_error_are_terminal() {
    for mode in [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 14] {
        let fixture = new_fixture();
        let recipe =
            ProviderRealBinaryRecipe::plan(fixture, request(ProviderRealBinaryOperation::Sub))
                .unwrap()
                .unwrap();
        let lhs = input(fixture);
        let rhs = input(fixture);
        fixture.mode.store(mode, Ordering::Relaxed);
        let error = ready(recipe.execute(&lhs.handle, &rhs.handle))
            .err()
            .unwrap();
        assert!(error.is_after_effect());
        if mode == 3 {
            assert!(std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .is_some());
        }
        lhs.handle.allocation_lease().validate_live().unwrap();
        rhs.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(lhs.probe.retirements(), 0);
        assert_eq!(rhs.probe.retirements(), 0);
        if !matches!(mode, 2 | 8) {
            assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_drop_preserves_both_borrowed_leases() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let fixture = new_fixture();
    let recipe = ProviderRealBinaryRecipe::plan(fixture, request(ProviderRealBinaryOperation::Div))
        .unwrap()
        .unwrap();
    let lhs = input(fixture);
    let rhs = input(fixture);
    fixture.mode.store(4, Ordering::Relaxed);
    let mut pending = Box::pin(recipe.execute(&lhs.handle, &rhs.handle));
    let waker = std::task::Waker::noop();
    assert!(matches!(
        pending.as_mut().poll(&mut Context::from_waker(waker)),
        Poll::Pending
    ));
    drop(pending);
    assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 1);
    assert_eq!(lhs.probe.retirements(), 0);
    assert_eq!(rhs.probe.retirements(), 0);
    lhs.handle.allocation_lease().validate_live().unwrap();
    rhs.handle.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn protected_third_allocation_is_never_retired_on_alias_rejection() {
    let fixture = new_fixture();
    let recipe = ProviderRealBinaryRecipe::plan(fixture, request(ProviderRealBinaryOperation::Sub))
        .unwrap()
        .unwrap();
    let lhs = input(fixture);
    let rhs = input(fixture);
    let public = input(fixture);
    *fixture.alias_target.lock().unwrap() = Some(public.handle.clone());
    fixture.mode.store(12, Ordering::Relaxed);
    let protected = [&public.handle];
    let error = ready(recipe.execute_protected(&lhs.handle, &rhs.handle, &protected))
        .err()
        .unwrap();
    assert!(error.is_after_effect());
    public.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(public.probe.retirements(), 0);
    assert_eq!(lhs.probe.retirements(), 0);
    assert_eq!(rhs.probe.retirements(), 0);

    let foreign_fixture = new_fixture();
    let foreign = input(foreign_fixture);
    *fixture.alias_target.lock().unwrap() = Some(foreign.handle.clone());
    fixture.mode.store(15, Ordering::Relaxed);
    let protected = [&foreign.handle];
    let error = ready(recipe.execute_protected(&lhs.handle, &rhs.handle, &protected))
        .err()
        .unwrap();
    assert!(error.is_after_effect());
    foreign.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(foreign.probe.retirements(), 0);
}
