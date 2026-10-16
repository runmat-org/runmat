//! API lifetime and validation seams, not WGPU resource/hardware evidence.
#[path = "cast_tests.rs"]
mod casts;
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
    protected_alias: Mutex<Option<GpuTensorHandle>>,
}
impl AccelProvider for Fixture {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "real recipe API lifetime fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no staging")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("no readback")
    }
    fn propose_real_elementwise(
        &self,
        request: &ProviderRealElementwiseRequest,
    ) -> ProviderExecutionResult<Option<ProviderRealElementwisePhysicalProposal>> {
        let (output, route) = output_projection(request);
        let alias = route == ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias;
        let mode = self.mode.load(Ordering::Relaxed);
        Ok(Some(ProviderRealElementwisePhysicalProposal {
            output_route: if mode == 11 {
                ProviderRealElementwiseOutputRoute::FreshAllocation
            } else {
                route
            },
            output_buffer_bytes: if alias {
                request.source_buffer_bytes
            } else {
                output.checked_byte_len().unwrap()
            },
            workgroup_size: 64,
            chunk_capacity: 64,
            uniform_allocations: if alias { u32::from(mode == 12) } else { 1 },
            uniform_buffer_bytes: if alias { 0 } else { 16 },
            dispatches: if alias { 0 } else { 1 },
            queue_submissions: if alias { 0 } else { 2 },
            pipeline_key: if alias { None } else { Some(42) },
        }))
    }
    fn execute_real_elementwise_recipe<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        recipe: &'a ProviderRealElementwiseRecipe,
        _admission: Option<&'a dyn runmat_execution::resource::ProviderHostMetadataAdmission>,
    ) -> crate::ProviderExecutionFuture<'a, ProviderExecutedRealElementwise> {
        Box::pin(async move {
            let mode = self.mode.load(Ordering::Relaxed);
            let output = if let Some(handle) = self.protected_alias.lock().unwrap().as_ref() {
                handle.with_view(handle.view().clone()).unwrap()
            } else if mode == 2
                || (recipe.physical().output_route
                    == ProviderRealElementwiseOutputRoute::SameClassFloatingCastAlias
                    && mode != 10)
            {
                input.with_view(input.view().clone()).unwrap()
            } else {
                let allocation = self
                    .owner
                    .allocate_zeroed(
                        if mode == 10 {
                            vec![1, 2, 1]
                        } else {
                            input.shape().to_vec()
                        },
                        if mode == 9 {
                            input.physical().element_type()
                        } else {
                            match recipe.output().element_type {
                                ProviderElementType::F32 => NumericElementType::F32,
                                ProviderElementType::F64 => NumericElementType::F64,
                                _ => unreachable!("fixture accepts real floating requests"),
                            }
                        },
                        GpuTensorStorage::Real,
                    )
                    .unwrap();
                self.produced.lock().unwrap().push(allocation.probe);
                allocation.handle
            };
            if mode == 3 {
                return Err(ProviderExecutionError::from(std::io::Error::other(
                    "original hook cause",
                ))
                .after_effect());
            }
            if mode == 4 {
                std::future::pending::<()>().await;
            }
            let p = recipe.physical();
            Ok(ProviderExecutedRealElementwise {
                observed: ProviderRealElementwiseObserved {
                    output_buffer_bytes: Some(
                        output.physical().requested_bytes() + u64::from(mode == 5),
                    ),
                    uniform_allocations: p.uniform_allocations + u32::from(mode == 8),
                    uniform_buffer_bytes: p.uniform_buffer_bytes + u64::from(mode == 1),
                    dispatches: p.dispatches + u32::from(mode == 6),
                    queue_submissions: p.queue_submissions + u32::from(mode == 7),
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
        protected_alias: Mutex::new(None),
    }));
    // SAFETY: leaked owner with a private distinct device; no ambient mutation.
    unsafe { crate::register_device_provider(fixture) };
    fixture
}
fn request(operation: ProviderRealElementwiseOperation) -> ProviderRealElementwiseRequest {
    ProviderRealElementwiseRequest {
        operation,
        source: ProviderRepresentation {
            element_type: ProviderElementType::F32,
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            shape: vec![2, 1, 1],
            residency: ProviderResidency::Device,
        },
        source_buffer_bytes: 8,
    }
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
fn original_scalar_class_bits_and_owner_are_not_wire_authority() {
    let fixture = new_fixture();
    let single = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::ScalarAdd {
            control: ScalarLiteral::F32Bits(1.0_f32.to_bits()),
            mode: runmat_types::RealFloatingAddMode::Single,
        }),
    )
    .unwrap()
    .unwrap();
    let signed_zero = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::ScalarAdd {
            control: ScalarLiteral::F32Bits((-0.0_f32).to_bits()),
            mode: runmat_types::RealFloatingAddMode::Single,
        }),
    )
    .unwrap()
    .unwrap();
    assert_ne!(single.identity().unwrap(), signed_zero.identity().unwrap());
    assert!(ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::ScalarAdd {
            control: ScalarLiteral::F64Bits(1.0_f64.to_bits()),
            mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
        })
    )
    .unwrap()
    .is_none());
    let foreign = new_fixture();
    assert!(!single.is_owned_by(foreign));
    let input = input(foreign);
    assert!(ready(single.execute(&input.handle)).is_err());
    assert_eq!(input.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn scalar_mul_max_physical_identity_preserves_operation_and_exact_control_bits() {
    let fixture = new_fixture();
    let mul = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::ScalarMul {
            control: ScalarLiteral::F32Bits(2.0_f32.to_bits()),
        }),
    )
    .unwrap()
    .unwrap();
    let max = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::ScalarMax {
            control: ScalarLiteral::F32Bits(2.0_f32.to_bits()),
        }),
    )
    .unwrap()
    .unwrap();
    assert_ne!(mul.identity().unwrap(), max.identity().unwrap());
    let negative_zero = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::ScalarMax {
            control: ScalarLiteral::F32Bits((-0.0_f32).to_bits()),
        }),
    )
    .unwrap()
    .unwrap();
    let positive_zero = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::ScalarMax {
            control: ScalarLiteral::F32Bits(0.0_f32.to_bits()),
        }),
    )
    .unwrap()
    .unwrap();
    assert_ne!(
        negative_zero.identity().unwrap(),
        positive_zero.identity().unwrap()
    );
    for operation in [
        ProviderRealElementwiseOperation::ScalarMul {
            control: ScalarLiteral::F64Bits(2.0_f64.to_bits()),
        },
        ProviderRealElementwiseOperation::ScalarMax {
            control: ScalarLiteral::F64Bits(2.0_f64.to_bits()),
        },
    ] {
        assert!(
            ProviderRealElementwiseRecipe::plan(fixture, request(operation))
                .unwrap()
                .is_none()
        );
    }
    let input = input(fixture);
    for recipe in [mul, max] {
        let result = ready(recipe.execute(&input.handle)).unwrap();
        assert_eq!(result.observed.output_buffer_bytes, Some(8));
        assert!(!result.output.shares_allocation_with(&input.handle));
    }
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn observed_mismatch_retires_only_owned_output_and_keeps_hook_cause() {
    for mode in [0, 1, 2, 3, 5, 6, 7, 8] {
        let fixture = new_fixture();
        fixture.mode.store(mode, Ordering::Relaxed);
        let recipe = ProviderRealElementwiseRecipe::plan(
            fixture,
            request(ProviderRealElementwiseOperation::Sqrt),
        )
        .unwrap()
        .unwrap();
        let input = input(fixture);
        let result = ready(recipe.execute(&input.handle));
        if mode == 0 {
            let result = result.unwrap();
            assert_eq!(result.observed.output_buffer_bytes, Some(8));
            assert!(!result.output.shares_allocation_with(&input.handle));
            drop(result);
        } else {
            let error = result.err().unwrap();
            assert!(error.is_after_effect());
            if mode == 3 {
                assert!(error.to_string().contains("original hook cause"));
                assert!(std::error::Error::source(&error)
                    .unwrap()
                    .downcast_ref::<std::io::Error>()
                    .is_some());
            }
        }
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        if mode != 2 {
            assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn cancellation_keeps_borrowed_input_and_releases_pending_owned_output() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    let fixture = new_fixture();
    fixture.mode.store(4, Ordering::Relaxed);
    let recipe = ProviderRealElementwiseRecipe::plan(
        fixture,
        request(ProviderRealElementwiseOperation::Sqrt),
    )
    .unwrap()
    .unwrap();
    let input = input(fixture);
    drop(Box::pin(recipe.execute(&input.handle)));
    assert!(
        fixture.produced.lock().unwrap().is_empty(),
        "unpolled cancellation has no provider effect"
    );
    let mut future = Box::pin(recipe.execute(&input.handle));
    let waker = std::task::Waker::noop();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(waker)),
        Poll::Pending
    ));
    assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 0);
    drop(future);
    assert_eq!(fixture.produced.lock().unwrap()[0].retirements(), 1);
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}
