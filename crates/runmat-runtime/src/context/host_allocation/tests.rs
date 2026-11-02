use std::rc::Rc;

use runmat_execution::{cancellation::CancellationSource, CancellationReason};
use runmat_value::{
    ExactHostAllocationPermit, ExactHostTensor2dAllocation, NumericDType, NumericScalar, Value,
};

use super::*;
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};

struct ManagedParallel(ManagedHostEffectLedger);

struct StaleReportedParallel(ManagedHostEffectLedger, u64);

impl RuntimeParallelService for StaleReportedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(self.0.snapshot().free_bytes()),
            epoch: self.1,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

impl RuntimeParallelService for ManagedParallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        let snapshot = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(snapshot.free_bytes()),
            epoch: snapshot.epoch,
            ..RuntimeParallelResources::default()
        }
    }

    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}

struct UnmanagedFinite;

#[test]
fn provider_control_admission_is_original_owner_bound_and_retained_until_last_alias() {
    use runmat_execution::resource::{ProviderHostMetadataAdmission, ProviderHostMetadataError};
    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let admission = ProviderHostControlAdmission::capture(Some(&context))
        .unwrap()
        .unwrap();
    assert_eq!(
        ledger.snapshot().live_reserved_bytes,
        0,
        "capture grants no bytes"
    );
    let record = admission.claim_record(128).unwrap();
    let bytes = HostAllocationPlan::retained_metadata_bytes(128).unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
    let alias = record.clone();
    drop(record);
    assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
    assert!(matches!(
        admission.claim_record(65_536),
        Err(ProviderHostMetadataError::ExceedsCapacity)
    ));
    assert_eq!(
        ledger.snapshot().live_reserved_bytes,
        bytes,
        "denial leaves original owner charged"
    );
    ledger.revoke();
    assert!(matches!(
        admission.claim_record(1),
        Err(ProviderHostMetadataError::StaleAuthority)
    ));
    assert_eq!(ledger.snapshot().live_reserved_bytes, bytes);
    drop(alias);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let missing = runtime(RuntimeServicePorts::default().with_parallel(Rc::new(UnmanagedFinite)));
    assert!(matches!(
        ProviderHostControlAdmission::capture(Some(&missing)),
        Err(HostAllocationError::MissingManagedOwner)
    ));
}

#[path = "tests/sparse_indexing.rs"]
mod sparse_indexing;

impl RuntimeParallelService for UnmanagedFinite {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }

    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources {
            memory_available_bytes: Some(64),
            ..RuntimeParallelResources::default()
        }
    }
}

fn runtime(ports: RuntimeServicePorts) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(ports)
}

#[test]
fn console_text_retains_one_backing_through_clones_and_rejects_partial_formatting() {
    use crate::console::ConsoleText;

    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let charged = ConsoleText::retained_bytes(5).unwrap();
    let receipt =
        HostAllocationReceipt::admit(Some(&context), HostAllocationPlan::new(0, charged).unwrap())
            .unwrap()
            .unwrap();
    let text =
        ConsoleText::try_initialize(5, &receipt.output_permit(&context).unwrap(), |writer| {
            writer.write_str("hello")
        })
        .unwrap();
    let shared = text.clone();
    assert_eq!(text.as_str().as_ptr(), shared.as_str().as_ptr());
    assert_eq!(text, "hello");
    assert!(shared.has_managed_backing());
    assert!(text.try_into_unbounded().is_err());
    drop(receipt);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(shared);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let receipt =
        HostAllocationReceipt::admit(Some(&context), HostAllocationPlan::new(0, charged).unwrap())
            .unwrap()
            .unwrap();
    let failure =
        ConsoleText::try_initialize(5, &receipt.output_permit(&context).unwrap(), |writer| {
            writer.write_str("hello")?;
            // Ignoring a failed write cannot publish the earlier prefix.
            let _ = writer.write_str("!");
            Ok(())
        });
    assert!(matches!(failure, Err(HostTextError::CapacityExceeded)));
    let formatter_failure =
        ConsoleText::try_initialize(0, &receipt.output_permit(&context).unwrap(), |_| {
            Err(std::fmt::Error)
        });
    assert!(matches!(
        formatter_failure,
        Err(HostTextError::FormattingFailed)
    ));
    assert_eq!(ConsoleText::retained_bytes(0).unwrap(), 0);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    for (contents, capacity) in [("short", 8192), ("", 64)] {
        let mut original = String::with_capacity(capacity);
        original.push_str(contents);
        let pointer = original.as_ptr();
        let source = ConsoleText::from(original);
        let charged = source.adoption_retained_bytes().unwrap();
        assert!(charged >= capacity as u64);
        let receipt = HostAllocationReceipt::admit(
            Some(&context),
            HostAllocationPlan::new(0, charged).unwrap(),
        )
        .unwrap()
        .unwrap();
        let adopted = source
            .try_adopt(&receipt.output_permit(&context).unwrap())
            .unwrap();
        assert_eq!(adopted.as_str(), contents);
        assert_eq!(adopted.as_str().as_ptr(), pointer);
        assert!(adopted.has_managed_backing());
        drop(receipt);
        assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
        drop(adopted);
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    let original = String::from("ordinary");
    let pointer = original.as_ptr();
    let ordinary = ConsoleText::from(original);
    assert_eq!(ordinary.as_str().as_ptr(), pointer);
    let ordinary = ordinary.try_into_unbounded().unwrap();
    assert_eq!(ordinary.as_ptr(), pointer);
}

#[test]
fn settled_scratch_handoff_respects_cancellation_and_retained_lifetime() {
    let ledger = ManagedHostEffectLedger::new_with_epoch(16, 5);
    let cancellation = CancellationSource::new();
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let receipt =
        HostAllocationReceipt::admit(Some(&context), HostAllocationPlan::new(16, 0).unwrap())
            .unwrap()
            .unwrap();
    receipt
        .transfer_settled_scratch_to_output(&context, 8)
        .expect("settled capacity transfer");
    let claim = receipt
        .output_permit(&context)
        .unwrap()
        .reserve_exact(8)
        .expect("retained backing");
    assert_eq!(ledger.snapshot().live_reserved_bytes, 16);
    cancellation.cancel(CancellationReason::User);
    assert_eq!(
        receipt.transfer_settled_scratch_to_output(&context, 1),
        Err(HostAllocationError::Cancelled)
    );
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 8);
    drop(claim);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn index_planner_retains_exact_owners_and_rejects_denial_or_cancel_before_callbacks() {
    use crate::indexing::plan::{
        build_expr_index_plan_in_context, build_index_plan_in_context, ExprPlanSpec,
    };
    use crate::indexing::selectors::SliceSelector;
    use crate::indexing::EndExpr;
    use std::cell::Cell;

    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let plan = build_index_plan_in_context(
        &context,
        &[SliceSelector::Colon, SliceSelector::Scalar(2)],
        2,
        &[3, 4],
    )
    .unwrap();
    assert_eq!(plan.indices, vec![3, 4, 5]);
    assert_eq!(plan.output_shape, vec![3, 1]);
    assert!(plan.indices.is_exact());
    assert!(
        plan.output_shape.is_admitted()
            && plan.selection_lengths.is_admitted()
            && plan.base_shape.is_admitted()
    );
    let charged = ledger.snapshot().live_claimed_bytes;
    assert!(charged > 0);
    let retained = plan.clone();
    assert_eq!(plan.indices.as_ptr(), retained.indices.as_ptr());
    assert_eq!(plan.output_shape.as_ptr(), retained.output_shape.as_ptr());
    assert_eq!(
        plan.selection_lengths.as_ptr(),
        retained.selection_lengths.as_ptr()
    );
    assert_eq!(plan.base_shape.as_ptr(), retained.base_shape.as_ptr());
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(plan);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(retained);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let calls = Cell::new(0usize);
    for reason in [
        None,
        Some(CancellationReason::User),
        Some(CancellationReason::Deadline),
    ] {
        let capacity = if reason.is_none() { 0 } else { 65_536 };
        let ledger = ManagedHostEffectLedger::new(capacity);
        let cancellation = CancellationSource::new();
        if let Some(reason) = reason {
            cancellation.cancel(reason);
        }
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        let result = futures::executor::block_on(build_expr_index_plan_in_context(
            &context,
            ExprPlanSpec {
                dims: 1,
                colon_mask: 0,
                end_mask: 0,
                range_dims: &[0],
                range_params: &[(1.0, 1.0)],
                range_start_exprs: &[None],
                range_step_exprs: &[None],
                range_end_exprs: &[EndExpr::End],
                numeric: &[],
                shape: &[1, 4],
            },
            |_, _| {
                calls.set(calls.get() + 1);
                std::future::ready(Ok(4.0))
            },
        ));
        assert!(result.is_err());
        assert_eq!(calls.get(), 0);
        if reason.is_some() {
            let expected = crate::execution::ExecutionServiceError::Cancelled
                .into_runtime_error_in_context(Some(&context));
            assert_eq!(result.unwrap_err().identifier(), expected.identifier());
        }
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }

    let failed = futures::executor::block_on(build_expr_index_plan_in_context(
        &context,
        ExprPlanSpec {
            dims: 1,
            colon_mask: 0,
            end_mask: 0,
            range_dims: &[0],
            range_params: &[(1.0, 1.0)],
            range_start_exprs: &[None],
            range_step_exprs: &[None],
            range_end_exprs: &[EndExpr::End],
            numeric: &[],
            shape: &[1, 4],
        },
        |_, _| {
            std::future::ready(Err(crate::runtime_error::semantic_error(
                "PlannerOriginal",
                "original end callback failure",
            )))
        },
    ));
    assert_eq!(
        failed.unwrap_err().identifier(),
        Some("RunMat:PlannerOriginal")
    );
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // The downstream gather keeps only its final payload, not a mirror of
    // the full input; its own explicit owner survives the planner's return.
    use crate::indexing::read_slice::{
        read_complex_slice_from_plan_in_context, read_tensor_slice_from_plan_in_context,
    };
    use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, Tensor};
    let selection =
        crate::indexing::plan::IndexPlan::new(vec![2, 0], vec![1, 2], vec![2], 1, vec![1, 3]);
    let tensor = Tensor::from_numeric_storage(
        runmat_value::NumericStorage::U64(vec![1, 2, u64::MAX]),
        vec![1, 3],
    )
    .unwrap();
    let complex = ComplexTensor::from_complex_storage(
        ComplexStorage::F32(
            vec![
                ComplexElement(f32::from_bits(0x7fc01234), -0.0),
                ComplexElement(2.0, 3.0),
                ComplexElement(4.0, 5.0),
            ]
            .into(),
        ),
        vec![1, 3],
    )
    .unwrap();
    let Value::Tensor(output) =
        read_tensor_slice_from_plan_in_context(Some(&context), &tensor, &selection).unwrap()
    else {
        panic!("typed tensor")
    };
    assert_eq!(
        output.numeric_value_at(0),
        Some(NumericScalar::U64(u64::MAX))
    );
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    let retained = output.clone();
    drop(output);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let Value::ComplexTensor(output) =
        read_complex_slice_from_plan_in_context(Some(&context), &complex, &selection).unwrap()
    else {
        panic!("complex tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    let (NumericScalar::F32(real), NumericScalar::F32(imag)) = output.numeric_value_at(1).unwrap()
    else {
        panic!("single components")
    };
    assert_eq!(real.to_bits(), 0x7fc01234);
    assert_eq!(imag.to_bits(), (-0.0f32).to_bits());
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    for reason in [
        None,
        Some(CancellationReason::User),
        Some(CancellationReason::Deadline),
    ] {
        let ledger = ManagedHostEffectLedger::new(if reason.is_none() { 0 } else { 65_536 });
        let cancellation = CancellationSource::new();
        if let Some(reason) = reason {
            cancellation.cancel(reason);
        }
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        for result in [
            read_tensor_slice_from_plan_in_context(Some(&context), &tensor, &selection),
            read_complex_slice_from_plan_in_context(Some(&context), &complex, &selection),
        ] {
            let error = result.expect_err("deny or cancel before output publication");
            if reason.is_some() {
                assert_eq!(
                    error.identifier(),
                    crate::execution::ExecutionServiceError::Cancelled
                        .into_runtime_error_in_context(Some(&context))
                        .identifier()
                );
            }
        }
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn indexed_deletion_retains_only_native_survivors_and_rejects_denial_or_cancel() {
    use crate::indexing::plan::IndexPlan;
    use crate::indexing::write_slice::{
        delete_complex_with_plan_in_context, delete_tensor_with_plan_in_context,
    };
    use runmat_value::{
        ComplexElement, ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage,
        Tensor,
    };
    let tensor =
        Tensor::new_integer(IntegerStorage::U64(vec![1, 2, u64::MAX]), vec![1, 3]).unwrap();
    let complex = ComplexTensor::from_complex_storage(
        ComplexStorage::F32(
            vec![
                ComplexElement(f32::from_bits(0x7fc01234), -0.0),
                ComplexElement(2.0, 3.0),
                ComplexElement(4.0, 5.0),
            ]
            .into(),
        ),
        vec![1, 3],
    )
    .unwrap();
    let paired = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1, 2, u64::MAX]),
            IntegerStorage::U64(vec![3, 4, u64::MAX]),
        )
        .unwrap(),
        vec![1, 3],
    )
    .unwrap();
    let delete = IndexPlan::new(vec![0, 1, 1], vec![1, 3], vec![3], 1, vec![1, 3]);
    let keep_first = IndexPlan::new(vec![1, 2, 2], vec![1, 3], vec![3], 1, vec![1, 3]);
    let empty = Value::Tensor(Tensor::zeros(vec![0, 0]));
    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let Value::Tensor(output) =
        delete_tensor_with_plan_in_context(Some(&context), tensor.clone(), &delete, &empty)
            .unwrap()
    else {
        panic!("deletion must not scalarize singleton")
    };
    assert_eq!(output.shape(), [1, 1]);
    assert_eq!(
        output.numeric_value_at(0),
        Some(NumericScalar::U64(u64::MAX))
    );
    assert_eq!(tensor.numeric_value_at(0), Some(NumericScalar::U64(1)));
    let retained = output.clone();
    drop(output);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let Value::ComplexTensor(output) =
        delete_complex_with_plan_in_context(Some(&context), complex.clone(), &keep_first, &empty)
            .unwrap()
    else {
        panic!("complex tensor")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    let (NumericScalar::F32(real), NumericScalar::F32(imag)) = output.numeric_value_at(0).unwrap()
    else {
        panic!("single complex")
    };
    assert_eq!(real.to_bits(), 0x7fc01234);
    assert_eq!(imag.to_bits(), (-0.0f32).to_bits());
    drop(output);
    let Value::ComplexTensor(output) =
        delete_complex_with_plan_in_context(Some(&context), paired, &delete, &empty).unwrap()
    else {
        panic!("integer complex tensor")
    };
    assert_eq!(
        output.numeric_value_at(0),
        Some((NumericScalar::U64(u64::MAX), NumericScalar::U64(u64::MAX)))
    );
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    {
        let _guard = context.enter();
        let Value::Tensor(output) = futures::executor::block_on(
            crate::indexing::write_linear::assign_tensor_scalar(tensor.clone(), &[2], &empty, true),
        )
        .unwrap() else {
            panic!("scalar deletion must retain tensor class")
        };
        assert_eq!(output.shape(), [1, 2]);
        assert_eq!(
            output.numeric_value_at(1),
            Some(NumericScalar::U64(u64::MAX))
        );
        drop(output);
        let Value::ComplexTensor(output) =
            futures::executor::block_on(crate::indexing::write_linear::assign_complex_scalar(
                complex.clone(),
                &[2],
                &empty,
                true,
            ))
            .unwrap()
        else {
            panic!("complex scalar deletion must retain tensor class")
        };
        let (NumericScalar::F32(real), NumericScalar::F32(imag)) =
            output.numeric_value_at(0).unwrap()
        else {
            panic!("single complex components")
        };
        assert_eq!(real.to_bits(), 0x7fc01234);
        assert_eq!(imag.to_bits(), (-0.0f32).to_bits());
        drop(output);
    }
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    for reason in [
        None,
        Some(CancellationReason::User),
        Some(CancellationReason::Deadline),
    ] {
        let ledger = ManagedHostEffectLedger::new(if reason.is_none() { 0 } else { 65_536 });
        let cancellation = CancellationSource::new();
        if let Some(reason) = reason {
            cancellation.cancel(reason);
        }
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        if reason.is_none() {
            let noop = IndexPlan::new(vec![], vec![1, 0], vec![0], 1, vec![1, 3]);
            let Value::Tensor(output) =
                delete_tensor_with_plan_in_context(Some(&context), tensor.clone(), &noop, &empty)
                    .unwrap()
            else {
                panic!("no-op tensor")
            };
            assert!(output
                .host_buffer()
                .shares_allocation_with(tensor.host_buffer()));
        }
        for result in [
            delete_tensor_with_plan_in_context(Some(&context), tensor.clone(), &delete, &empty),
            delete_complex_with_plan_in_context(
                Some(&context),
                complex.clone(),
                &keep_first,
                &empty,
            ),
        ] {
            let error = result.expect_err("deny or cancel before deletion publication");
            if reason.is_some() {
                assert_eq!(
                    error.identifier(),
                    crate::execution::ExecutionServiceError::Cancelled
                        .into_runtime_error_in_context(Some(&context))
                        .identifier()
                );
            }
        }
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn dense_planned_assignment_reuses_unique_owners_and_funds_only_required_copies() {
    use crate::indexing::plan::IndexPlan;
    use crate::indexing::write_slice::{
        assign_complex_with_plan_host_in_context, assign_tensor_with_plan_host_in_context,
    };
    use runmat_value::{
        ComplexElement, ComplexStorage, ComplexTensor, IntegerComplexStorage, IntegerStorage,
        Tensor,
    };
    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let source =
        Tensor::from_f32(vec![f32::from_bits(0x7fc01234), -0.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let alias = source.clone();
    let rhs =
        Value::Tensor(Tensor::from_f32(vec![f32::from_bits(0x7fc05678)], vec![1, 1]).unwrap());
    let growth = IndexPlan::new(vec![2, 5], vec![1, 2], vec![1, 2], 2, vec![3, 2]);
    let Value::Tensor(grown) =
        assign_tensor_with_plan_host_in_context(Some(&context), source, &growth, &rhs).unwrap()
    else {
        panic!("single tensor")
    };
    let values = grown.as_f32_slice().unwrap();
    assert_eq!(grown.shape(), [3, 2]);
    assert_eq!(values[0].to_bits(), 0x7fc01234);
    assert_eq!(values[1].to_bits(), (-0.0f32).to_bits());
    assert_eq!(values[2].to_bits(), 0x7fc05678);
    assert_eq!(&values[3..5], &[3.0, 4.0]);
    assert_eq!(values[5].to_bits(), 0x7fc05678);
    assert_eq!(alias.shape(), [2, 2]);
    let pointer = grown.as_f32_slice().unwrap().as_ptr();
    let charged = ledger.snapshot().live_claimed_bytes;
    let repeated = IndexPlan::new(vec![1, 1], vec![1, 2], vec![2], 1, vec![3, 2]);
    let rhs =
        Value::Tensor(Tensor::from_f32(vec![2.0, f32::from_bits(0x7fc07890)], vec![1, 2]).unwrap());
    let Value::Tensor(updated) =
        assign_tensor_with_plan_host_in_context(Some(&context), grown, &repeated, &rhs).unwrap()
    else {
        panic!("unique tensor")
    };
    assert_eq!(updated.as_f32_slice().unwrap().as_ptr(), pointer);
    assert_eq!(updated.as_f32_slice().unwrap()[1].to_bits(), 0x7fc07890);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(updated);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let complex = ComplexTensor::from_complex_storage(
        ComplexStorage::F32(vec![ComplexElement(1.0, 2.0)].into()),
        vec![1, 1],
    )
    .unwrap();
    let complex_alias = complex.clone();
    let pointer = complex.as_f32_slice().unwrap().as_ptr();
    let singleton = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 1]);
    let rhs = Value::ComplexTensor(
        ComplexTensor::from_complex_storage(
            ComplexStorage::F32(vec![ComplexElement(f32::from_bits(0x7fc01234), -0.0)].into()),
            vec![1, 1],
        )
        .unwrap(),
    );
    let Value::ComplexTensor(updated) =
        assign_complex_with_plan_host_in_context(Some(&context), complex, &singleton, &rhs)
            .unwrap()
    else {
        panic!("unique complex")
    };
    assert_ne!(updated.as_f32_slice().unwrap().as_ptr(), pointer);
    assert_eq!(updated.as_f32_slice().unwrap()[0].0.to_bits(), 0x7fc01234);
    assert_eq!(
        updated.as_f32_slice().unwrap()[0].1.to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(
        complex_alias.as_f32_slice().unwrap()[0],
        ComplexElement(1.0, 2.0)
    );
    let pointer = updated.as_f32_slice().unwrap().as_ptr();
    let charged = ledger.snapshot().live_claimed_bytes;
    let Value::ComplexTensor(updated) =
        assign_complex_with_plan_host_in_context(Some(&context), updated, &singleton, &rhs)
            .unwrap()
    else {
        panic!("unique retained complex")
    };
    assert_eq!(updated.as_f32_slice().unwrap().as_ptr(), pointer);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(updated);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let source = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
    let promotion = IndexPlan::new(vec![1], vec![1, 1], vec![1], 1, vec![1, 2]);
    let rhs = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                IntegerStorage::U64(vec![u64::MAX]),
            )
            .unwrap(),
            vec![1, 1],
        )
        .unwrap(),
    );
    let Value::ComplexTensor(output) =
        assign_tensor_with_plan_host_in_context(Some(&context), source, &promotion, &rhs).unwrap()
    else {
        panic!("promoted exact complex")
    };
    assert_eq!(
        output.numeric_value_at(0),
        Some((NumericScalar::U64(u64::MAX), NumericScalar::U64(0)))
    );
    assert_eq!(
        output.numeric_value_at(1),
        Some((
            NumericScalar::U64(9_007_199_254_740_993),
            NumericScalar::U64(u64::MAX)
        ))
    );
    let pointer = match output.integer_storage().unwrap().real() {
        IntegerStorage::U64(values) => values.as_ptr(),
        _ => unreachable!("uint64 target"),
    };
    let charged = ledger.snapshot().live_claimed_bytes;
    let update = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 2]);
    let Value::ComplexTensor(output) = assign_complex_with_plan_host_in_context(
        Some(&context),
        output,
        &update,
        &Value::Complex(2.6, -2.6),
    )
    .unwrap() else {
        panic!("unique exact integer complex")
    };
    let retained_pointer = match output.integer_storage().unwrap().real() {
        IntegerStorage::U64(values) => values.as_ptr(),
        _ => unreachable!("uint64 target"),
    };
    assert_eq!(retained_pointer, pointer);
    assert_eq!(
        output.numeric_value_at(0),
        Some((NumericScalar::U64(3), NumericScalar::U64(0)))
    );
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(output);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    for reason in [
        None,
        Some(CancellationReason::User),
        Some(CancellationReason::Deadline),
    ] {
        let ledger = ManagedHostEffectLedger::new(if reason.is_none() { 0 } else { 65_536 });
        let cancellation = CancellationSource::new();
        if let Some(reason) = reason {
            cancellation.cancel(reason);
        }
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        let source = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let alias = source.clone();
        let error = assign_tensor_with_plan_host_in_context(
            Some(&context),
            source,
            &singleton,
            &Value::Num(2.0),
        )
        .expect_err("shared output requires funding or cancellation");
        if reason.is_some() {
            assert_eq!(
                error.identifier(),
                crate::execution::ExecutionServiceError::Cancelled
                    .into_runtime_error_in_context(Some(&context))
                    .identifier()
            );
        }
        assert_eq!(alias.as_f32_slice().unwrap(), &[1.0]);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        if reason.is_none() {
            let unique = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
            let pointer = unique.as_f32_slice().unwrap().as_ptr();
            let Value::Tensor(output) = assign_tensor_with_plan_host_in_context(
                Some(&context),
                unique,
                &singleton,
                &Value::Num(2.0),
            )
            .unwrap() else {
                panic!("unique zero-budget update")
            };
            assert_eq!(output.as_f32_slice().unwrap().as_ptr(), pointer);
            assert_eq!(output.as_f32_slice().unwrap(), &[2.0]);
        }
    }
}

#[test]
fn dense_scalar_assignment_retains_unique_and_funds_growth_or_promotion() {
    use crate::indexing::write_linear::{
        assign_complex_scalar_in_context, assign_tensor_scalar_in_context,
    };
    use futures::executor::block_on;
    use runmat_value::{ComplexTensor, IntegerComplexStorage, IntegerStorage, Tensor};
    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let source = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap();
    let alias = source.clone();
    let Value::Tensor(grown) = block_on(assign_tensor_scalar_in_context(
        Some(&context),
        source,
        &[3],
        &Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
        false,
    ))
    .unwrap() else {
        panic!("integer growth")
    };
    assert_eq!(grown.shape(), [1, 3]);
    assert_eq!(
        grown.numeric_value_at(0),
        Some(NumericScalar::U64(u64::MAX))
    );
    assert_eq!(grown.numeric_value_at(1), Some(NumericScalar::U64(0)));
    assert_eq!(
        grown.numeric_value_at(2),
        Some(NumericScalar::U64(9_007_199_254_740_993))
    );
    assert_eq!(
        alias.numeric_value_at(0),
        Some(NumericScalar::U64(u64::MAX))
    );
    drop(grown);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let rhs = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                IntegerStorage::U64(vec![u64::MAX]),
            )
            .unwrap(),
            vec![1, 1],
        )
        .unwrap(),
    );
    let Value::ComplexTensor(promoted) = block_on(assign_tensor_scalar_in_context(
        Some(&context),
        alias,
        &[3],
        &rhs,
        false,
    ))
    .unwrap() else {
        panic!("direct promoted final owner")
    };
    assert_eq!(promoted.shape(), [1, 3]);
    assert_eq!(
        promoted.numeric_value_at(0),
        Some((NumericScalar::U64(u64::MAX), NumericScalar::U64(0)))
    );
    assert_eq!(
        promoted.numeric_value_at(1),
        Some((NumericScalar::U64(0), NumericScalar::U64(0)))
    );
    assert_eq!(
        promoted.numeric_value_at(2),
        Some((
            NumericScalar::U64(9_007_199_254_740_993),
            NumericScalar::U64(u64::MAX)
        ))
    );
    let pointer = match promoted.integer_storage().unwrap().real() {
        IntegerStorage::U64(values) => values.as_ptr(),
        _ => unreachable!("uint64 class"),
    };
    let charged = ledger.snapshot().live_claimed_bytes;
    let Value::ComplexTensor(updated) = block_on(assign_complex_scalar_in_context(
        Some(&context),
        promoted,
        &[1],
        &Value::Complex(2.6, -2.6),
        false,
    ))
    .unwrap() else {
        panic!("unique retained scalar complex")
    };
    assert_eq!(
        updated.numeric_value_at(0),
        Some((NumericScalar::U64(3), NumericScalar::U64(0)))
    );
    assert_eq!(
        match updated.integer_storage().unwrap().real() {
            IntegerStorage::U64(values) => values.as_ptr(),
            _ => unreachable!("uint64 class"),
        },
        pointer
    );
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
    drop(updated);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    for reason in [
        None,
        Some(CancellationReason::User),
        Some(CancellationReason::Deadline),
    ] {
        let ledger = ManagedHostEffectLedger::new(if reason.is_none() { 0 } else { 65_536 });
        let cancellation = CancellationSource::new();
        if let Some(reason) = reason {
            cancellation.cancel(reason);
        }
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        )
        .with_cancellation_token(cancellation.token());
        let source = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let alias = source.clone();
        let error = block_on(assign_tensor_scalar_in_context(
            Some(&context),
            source,
            &[1],
            &Value::Num(2.0),
            false,
        ))
        .expect_err("copy denied or cancelled");
        if reason.is_some() {
            assert_eq!(
                error.identifier(),
                crate::execution::ExecutionServiceError::Cancelled
                    .into_runtime_error_in_context(Some(&context))
                    .identifier()
            );
        }
        assert_eq!(alias.as_f32_slice().unwrap(), &[1.0]);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        if reason.is_none() {
            let source = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let pointer = source.as_f32_slice().unwrap().as_ptr();
            let rhs = Value::Tensor(
                Tensor::from_f32(vec![f32::from_bits(0x7fc01234)], vec![1, 1]).unwrap(),
            );
            let Value::Tensor(output) = block_on(assign_tensor_scalar_in_context(
                Some(&context),
                source,
                &[1, 99],
                &rhs,
                false,
            ))
            .unwrap() else {
                panic!("unique clamped subscript")
            };
            assert_eq!(output.shape(), [2, 2]);
            assert_eq!(output.as_f32_slice().unwrap().as_ptr(), pointer);
            assert_eq!(output.as_f32_slice().unwrap()[2].to_bits(), 0x7fc01234);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
    }
}

#[test]
fn single_assignment_uses_canonical_direct_wide_integer_cast_on_every_owner() {
    use crate::indexing::plan::IndexPlan;
    use crate::indexing::write_linear::{
        assign_complex_scalar_in_context, assign_tensor_scalar_in_context,
    };
    use crate::indexing::write_slice::{
        assign_complex_with_plan_host_in_context, assign_tensor_with_plan_host_in_context,
    };
    use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, IntValue, Tensor};
    let witness = (1u64 << 63) + (1u64 << 39) + 1;
    let rhs = Value::Int(IntValue::U64(witness));
    let plan = IndexPlan::new(vec![0], vec![1, 1], vec![1], 1, vec![1, 1]);
    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    for shared in [false, true] {
        for scalar in [false, true] {
            let tensor = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
            let alias = shared.then(|| tensor.clone());
            let result = if scalar {
                futures::executor::block_on(assign_tensor_scalar_in_context(
                    Some(&context),
                    tensor,
                    &[1],
                    &rhs,
                    false,
                ))
            } else {
                assign_tensor_with_plan_host_in_context(Some(&context), tensor, &plan, &rhs)
            };
            let Value::Tensor(output) = result.unwrap() else {
                panic!("native single result")
            };
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
            assert_eq!(output.as_f32_slice().unwrap()[0].to_bits(), 0x5f000001);
            if let Some(alias) = alias {
                assert_eq!(alias.as_f32_slice().unwrap(), &[0.0]);
            }
            drop(output);
            let complex = ComplexTensor::from_complex_storage(
                ComplexStorage::F32(vec![ComplexElement(0.0, 0.0)].into()),
                vec![1, 1],
            )
            .unwrap();
            let alias = shared.then(|| complex.clone());
            let result = if scalar {
                futures::executor::block_on(assign_complex_scalar_in_context(
                    Some(&context),
                    complex,
                    &[1],
                    &rhs,
                    false,
                ))
            } else {
                assign_complex_with_plan_host_in_context(Some(&context), complex, &plan, &rhs)
            };
            let Value::ComplexTensor(output) = result.unwrap() else {
                panic!("native single complex result")
            };
            assert_eq!(output.numeric_dtype(), NumericDType::F32);
            assert_eq!(output.as_f32_slice().unwrap()[0].0.to_bits(), 0x5f000001);
            if let Some(alias) = alias {
                assert_eq!(alias.as_f32_slice().unwrap()[0], ComplexElement(0.0, 0.0));
            }
            drop(output);
            assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
        }
        let mut tensor = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
        let alias = shared.then(|| tensor.clone());
        tensor
            .set_numeric_assignment_at(0, NumericScalar::U64(witness))
            .unwrap();
        assert_eq!(tensor.as_f32_slice().unwrap()[0].to_bits(), 0x5f000001);
        if let Some(alias) = alias {
            assert_eq!(alias.as_f32_slice().unwrap(), &[0.0]);
        }
    }
    assert_eq!(rhs, Value::Int(IntValue::U64(witness)));
}

#[test]
fn trace_scratch_growth_claims_exact_live_bytes_and_revokes_new_growth() {
    let ledger = ManagedHostEffectLedger::new_with_epoch(4_096, 7);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let scratch = HostAllocationReceipt::dynamic_trace_scratch(&runtime)
        .unwrap()
        .expect("finite trace issuer");
    let claim = scratch.reserve_exact_claim(128).expect("admitted growth");
    let charged = 128 + crate::context::host_effect::owned_output_claim_metadata_bytes();
    assert_eq!(ledger.snapshot().live_reserved_bytes, charged as u64);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged as u64);

    cancellation.cancel(CancellationReason::User);
    assert!(scratch.reserve_exact_claim(1).is_err());
    assert_eq!(
        scratch.source_failure(runmat_value::ExactHostNumericAllocationError::ExceedsPermit),
        HostAllocationError::Cancelled
    );
    drop(claim);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn control_backing_keeps_exact_claim_through_iteration_and_owner_cancellation() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HostControlAllocationOwner>();
    let ledger = ManagedHostEffectLedger::new_with_epoch(4_096, 7);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let owner = HostControlAllocationOwner::capture(&runtime)
        .unwrap()
        .unwrap();
    let owner = std::thread::spawn(move || {
        // No RuntimeContext or thread-affine service crosses this boundary.
        let claim = owner.claim(19).unwrap();
        drop(claim);
        owner
    })
    .join()
    .unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    owner.checkpoint(&runtime).unwrap();
    struct PendingControl {
        ledger: ManagedHostEffectLedger,
        _bytes: [u8; 8],
    }
    impl std::future::Future for PendingControl {
        type Output = ();
        fn poll(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<()> {
            std::task::Poll::Pending
        }
    }
    impl Drop for PendingControl {
        fn drop(&mut self) {
            assert!(self.ledger.snapshot().live_claimed_bytes > 0);
        }
    }
    let control = HostControlFuture::new(Some(&owner), || PendingControl {
        ledger: ledger.clone(),
        _bytes: [7; 8],
    })
    .unwrap();
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        (std::mem::size_of::<PendingControl>()
            + crate::context::host_effect::owned_output_claim_metadata_bytes()) as u64
    );
    drop(control);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    let mut values = HostControlVec::<u64>::with_capacity(2, Some(&owner)).unwrap();
    values.try_push(13).unwrap();
    values.try_push(29).unwrap();
    assert_eq!(values.try_push(41), Err(41));
    let header = crate::context::host_effect::owned_output_claim_metadata_bytes();
    assert_eq!(ledger.snapshot().live_claimed_bytes, (16 + header) as u64);
    values.reserve_capacity(4).unwrap();
    values.try_push(41).unwrap();
    let charged = 32 + header;
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged as u64);
    assert_eq!(
        ledger.snapshot().peak_claimed_bytes,
        (16 + 32 + 2 * header) as u64
    );
    let mut iteration = values.into_iter();
    assert_eq!(iteration.next(), Some(13));
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        HostControlFuture::<PendingControl>::new(Some(&owner), || {
            panic!("revoked owner cannot construct a process supervisor")
        }),
        Err(HostAllocationError::Cancelled)
    ));
    assert!(matches!(
        owner.claim(1),
        Err(HostAllocationError::Cancelled)
    ));
    drop(owner);
    assert_eq!(ledger.snapshot().live_claimed_bytes, charged as u64);
    assert_eq!(iteration.next(), Some(29));
    assert_eq!(iteration.next(), Some(41));
    drop(iteration);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn admitted_collection_blocks_survive_retirement_without_new_growth() {
    use allocator_api2::alloc::Allocator;
    use std::alloc::Layout;

    for layout in [
        Layout::from_size_align(37, 64).unwrap(),
        Layout::from_size_align(0, 64).unwrap(),
    ] {
        let ledger = ManagedHostEffectLedger::new(65_536);
        let context = runtime(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
        let physical = AdmittedCollectionAllocator::requested_bytes(layout).unwrap();
        let charged = HostAllocationPlan::retained_exact_output_bytes(
            physical as u64,
            AdmittedCollectionAllocator::retained_claim_count(),
        )
        .unwrap();
        let receipt = HostAllocationReceipt::admit(
            Some(&context),
            HostAllocationPlan::new(0, charged).unwrap(),
        )
        .unwrap()
        .unwrap();
        let allocator = AdmittedCollectionAllocator::new(receipt.output_permit(&context).unwrap());
        let block = allocator.allocate(layout).unwrap();
        let pointer = block.cast::<u8>();
        assert_eq!(pointer.as_ptr().align_offset(layout.align()), 0);
        // SAFETY: the returned block has exactly the requested writable size.
        unsafe { pointer.as_ptr().write_bytes(0x5a, layout.size()) };
        drop(receipt);
        assert_eq!(ledger.snapshot().live_claimed_bytes, charged);
        assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
        let replacement = Layout::from_size_align(layout.size() + 1, layout.align()).unwrap();
        // SAFETY: pointer/layout identify the live block. Failed growth must
        // leave it owned by the original allocation, even after revocation.
        assert!(unsafe { allocator.clone().grow(pointer, layout, replacement) }.is_err());
        for index in 0..layout.size() {
            // SAFETY: the original block remains live after failed growth.
            assert_eq!(unsafe { pointer.as_ptr().add(index).read() }, 0x5a);
        }
        // SAFETY: consume the sole block with its original exact layout.
        unsafe { allocator.deallocate(pointer, layout) };
        assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}

#[test]
fn operation_collection_growth_uses_only_the_installed_receipt() {
    use allocator_api2::alloc::Allocator;
    use std::alloc::Layout;

    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let layout = Layout::from_size_align(41, 16).unwrap();
    let block = HostAllocationPlan::retained_exact_output_bytes(
        AdmittedCollectionAllocator::requested_bytes(layout).unwrap() as u64,
        AdmittedCollectionAllocator::retained_claim_count(),
    )
    .unwrap();
    let first = HostAllocationReceipt::admit(
        Some(&context),
        HostAllocationPlan::new(0, OperationPermitGate::retained_bytes().unwrap() + block).unwrap(),
    )
    .unwrap()
    .unwrap();
    let gate = OperationPermitGate::prepare(&first.output_permit(&context).unwrap()).unwrap();
    let allocator = OperationCollectionAllocator::new(gate.clone());
    assert!(allocator.allocate(layout).is_err());
    let installed = gate
        .install(first.output_permit(&context).unwrap())
        .unwrap();
    assert!(matches!(
        gate.install(first.output_permit(&context).unwrap()),
        Err(OperationPermitGateError::AlreadyInstalled)
    ));
    let first_block = allocator.allocate(layout).unwrap().cast::<u8>();
    drop(installed);
    drop(first);
    assert!(allocator.allocate(layout).is_err());

    let second =
        HostAllocationReceipt::admit(Some(&context), HostAllocationPlan::new(0, block).unwrap())
            .unwrap()
            .unwrap();
    let installed = gate
        .install(second.output_permit(&context).unwrap())
        .unwrap();
    let second_block = allocator.allocate(layout).unwrap().cast::<u8>();
    drop(installed);
    drop(second);
    unsafe {
        allocator.deallocate(first_block, layout);
        allocator.deallocate(second_block, layout);
    }
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        OperationPermitGate::retained_bytes().unwrap()
    );
    drop(allocator);
    drop(gate);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn session_random_state_claim_survives_snapshot_and_rehomes_before_cow() {
    use crate::builtins::common::random::RandomState;

    let ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 19);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let mut state = RandomState::seeded(41);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    state.prepare_mutation(Some(&runtime)).unwrap();
    let one_state = ledger.snapshot().live_claimed_bytes;
    assert!(one_state > 624 * 4);
    let saved = state.clone();
    state.advance_uniforms(1, Some(&runtime)).unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, 2 * one_state);

    let next_ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 20);
    let rebound = runtime.clone().with_service_ports(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(next_ledger.clone()))),
    );
    state.advance_uniforms(1, Some(&rebound)).unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, one_state);
    assert_eq!(next_ledger.snapshot().live_claimed_bytes, one_state);
    drop(saved);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop(state);
    assert_eq!(next_ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn session_random_state_rejects_missing_owner_and_cancel_before_claim() {
    use crate::builtins::common::random::RandomState;

    let unmanaged = runtime(RuntimeServicePorts::default().with_parallel(Rc::new(UnmanagedFinite)));
    let mut state = RandomState::seeded(41);
    assert!(state.prepare_mutation(Some(&unmanaged)).is_err());
    assert_eq!(state, RandomState::seeded(41));

    let ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 21);
    let cancellation = CancellationSource::new();
    let cancelled = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    cancellation.cancel(CancellationReason::User);
    assert!(state.prepare_mutation(Some(&cancelled)).is_err());
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn retained_state_rechecks_ledgers_epoch_even_if_service_reports_stale_epoch() {
    use crate::builtins::common::random::RandomState;

    let ledger = ManagedHostEffectLedger::new_with_epoch(16_384, 27);
    let runtime = runtime(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(StaleReportedParallel(ledger.clone(), 27))),
    );
    let mut state = RandomState::seeded(41);
    state.prepare_mutation(Some(&runtime)).unwrap();
    let live = ledger.snapshot().live_claimed_bytes;
    ledger.set_epoch_for_test(28);
    assert!(state.prepare_mutation(Some(&runtime)).is_err());
    assert_eq!(ledger.snapshot().live_claimed_bytes, live);
    drop(state);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn ordinary_numeric_call_shares_one_ledger_and_transfers_retained_output() {
    let scratch = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, 4).unwrap();
    let output = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 4]).unwrap();
    let plan = HostAllocationPlan::new(scratch, output).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(scratch + output, 7);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .unwrap()
        .unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, scratch + output);
    let mut workspace = receipt
        .temporary_numeric(&runtime, NumericDType::F64, 4)
        .unwrap();
    workspace
        .as_f64_slice_mut()
        .unwrap()
        .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ledger.snapshot().live_claimed_bytes, scratch);
    drop(workspace);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let mut native_samples = receipt
        .temporary_numeric(&runtime, NumericDType::I16, 4)
        .unwrap();
    native_samples.set_value(0, NumericScalar::I16(17)).unwrap();
    assert!(matches!(
        native_samples.data_view(),
        runmat_accelerate_api::HostNumericDataView::I16(values) if values == [17, 0, 0, 0]
    ));
    drop(native_samples);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let tensor = receipt
        .with_output_allocation(&runtime, |permit| {
            let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 4, permit)?;
            allocation
                .as_f64_slice_mut()
                .unwrap()
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            allocation.into_tensor()
        })
        .unwrap();
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0]);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, output);
    assert_eq!(ledger.snapshot().live_claimed_bytes, output);
    drop(tensor);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    // Character conversion retains exactly the same final double owner.
    let chars = runmat_value::CharArray::from_code_units(
        [65, 233, 0xD800, 4]
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .to_vec(),
        2,
        2,
    )
    .unwrap();
    let converted =
        crate::builtins::common::tensor::character_numeric::to_double(Some(&runtime), &chars)
            .unwrap();
    assert_eq!(converted.shape(), [2, 2]);
    assert_eq!(converted.materialize_f64(), vec![65., 55296., 233., 4.]);
    assert_eq!(ledger.snapshot().live_claimed_bytes, output);
    drop(converted);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn complex_scratch_uses_one_claimed_interleaved_real_backing() {
    let scratch = HostAllocationPlan::complex_workspace_bytes(3).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(scratch, 7);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt =
        HostAllocationReceipt::admit(Some(&runtime), HostAllocationPlan::new(scratch, 0).unwrap())
            .unwrap()
            .unwrap();
    let mut workspace = receipt.temporary_complex(&runtime, 3).unwrap();
    assert_eq!(workspace.len(), 3);
    assert_eq!(
        workspace.get(2),
        Some(num_complex::Complex64::new(0.0, 0.0))
    );
    assert_eq!(
        workspace.set(2, num_complex::Complex64::new(4.0, -1.0)),
        Some(())
    );
    assert_eq!(
        workspace.get(2),
        Some(num_complex::Complex64::new(4.0, -1.0))
    );
    assert_eq!(workspace.get(3), None);
    assert_eq!(ledger.snapshot().live_claimed_bytes, scratch);
    drop(workspace);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);

    let mut lanes = receipt
        .temporary_numeric(&runtime, NumericDType::F64, 6)
        .unwrap();
    lanes.as_complex_f64_slice_mut().unwrap()[1] = runmat_value::ComplexElement(2.0, -3.0);
    assert_eq!(lanes.as_f64_slice().unwrap()[2..4], [2.0, -3.0]);
    drop(lanes);
    let mut singles = receipt
        .temporary_numeric(&runtime, NumericDType::F32, 6)
        .unwrap();
    singles.as_complex_f32_slice_mut().unwrap()[2] = runmat_value::ComplexElement(4.0, -5.0);
    assert_eq!(singles.as_f32_slice_mut().unwrap()[4..6], [4.0, -5.0]);
    drop(singles);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
}

#[test]
fn builtin_continuation_keeps_box_claim_after_ready_and_rechecks_before_poll() {
    let ledger = ManagedHostEffectLedger::new_with_epoch(1 << 20, 19);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    runtime.set_error_namespace("ContinuationScope");

    let polled = Rc::new(std::cell::Cell::new(false));
    let observed = polled.clone();
    let mut pending =
        crate::builtin::admitted_builtin_future(Some(&runtime), "continuation_test", async move {
            observed.set(true);
            Ok(crate::execution::RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::Num(4.0)).unwrap(),
            )
            .unwrap())
        });
    let charged = ledger.snapshot().live_reserved_bytes;
    assert!(charged > 0);
    cancellation.cancel(CancellationReason::User);
    let error = futures::executor::block_on(pending.as_mut()).unwrap_err();
    assert_eq!(error.identifier(), Some("ContinuationScope:Cancelled"));
    assert!(
        !polled.get(),
        "revoked continuation must not enter builtin body"
    );
    assert!(error
        .source
        .as_deref()
        .and_then(|source| source.downcast_ref::<crate::execution::ExecutionServiceError>())
        .is_some_and(|source| matches!(
            source,
            crate::execution::ExecutionServiceError::Cancelled
        )));
    assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
    drop(pending);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let fresh = runtime.with_cancellation_token(CancellationSource::new().token());
    let mut completed =
        crate::builtin::admitted_builtin_future(Some(&fresh), "continuation_test", async {
            Ok(crate::execution::RootedValueSequence::new(
                runmat_value::ValueSequence::single(Value::Num(5.0)).unwrap(),
            )
            .unwrap())
        });
    let charged = ledger.snapshot().live_reserved_bytes;
    assert!(charged > 0);
    let output = futures::executor::block_on(completed.as_mut()).unwrap();
    assert_eq!(ledger.snapshot().live_reserved_bytes, charged);
    drop(output);
    drop(completed);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    // An unbounded invocation has no allocation receipt, but still owns its
    // cancellation boundary both before entry and after a successful poll.
    for cancel_before_poll in [true, false] {
        let source = CancellationSource::new();
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_cancellation_token(source.token());
        runtime.set_error_namespace("UnboundedContinuation");
        let polled = Rc::new(std::cell::Cell::new(false));
        let observed = polled.clone();
        let inside = source.clone();
        let pending = crate::builtin::admitted_builtin_future(
            Some(&runtime),
            "continuation_test",
            async move {
                observed.set(true);
                inside.cancel(CancellationReason::User);
                Ok(crate::execution::RootedValueSequence::new(
                    runmat_value::ValueSequence::single(Value::Num(6.0)).unwrap(),
                )
                .unwrap())
            },
        );
        if cancel_before_poll {
            source.cancel(CancellationReason::User);
        }
        let error = futures::executor::block_on(pending).unwrap_err();
        assert_eq!(polled.get(), !cancel_before_poll);
        assert_eq!(error.identifier(), Some("UnboundedContinuation:Cancelled"));
    }
}

#[test]
fn owned_cpu_work_retains_scratch_after_cancelled_caller_retires_receipt() {
    let scratch = HostAllocationPlan::numeric_workspace_bytes(NumericDType::F64, 4).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(scratch, 13);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt =
        HostAllocationReceipt::admit(Some(&runtime), HostAllocationPlan::new(scratch, 0).unwrap())
            .unwrap()
            .unwrap();
    let retained = receipt.retain_scratch_for_cpu_work();
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, scratch);
    drop(retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn output_allocation_restores_control_cause_and_drops_unpublished_claim() {
    let output = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 2]).unwrap();
    let plan = HostAllocationPlan::new(0, output).unwrap();
    let ledger = ManagedHostEffectLedger::new_with_epoch(output, 11);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .unwrap()
        .unwrap();
    let result = receipt.with_output_allocation(&runtime, |permit| {
        let mut allocation = ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 2, permit)?;
        allocation
            .as_f64_slice_mut()
            .unwrap()
            .copy_from_slice(&[3.0, 4.0]);
        cancellation.cancel(CancellationReason::User);
        allocation.into_tensor()
    });
    assert!(matches!(result, Err(HostAllocationError::Cancelled)));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    let fresh = runtime.with_cancellation_token(CancellationSource::new().token());
    let receipt = HostAllocationReceipt::admit(Some(&fresh), plan)
        .unwrap()
        .unwrap();
    let result = receipt.with_output_allocation(&fresh, |permit| {
        fresh.invalidate_host_effect_session_authority();
        ExactHostTensor2dAllocation::new(NumericDType::F64, 1, 2, permit)
    });
    assert!(matches!(result, Err(HostAllocationError::StaleAuthority)));
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn retained_view_claim_outlives_stage_receipt_without_holding_scratch() {
    let view = 80;
    let per_view = HostAllocationPlan::retained_view_bytes(view).unwrap();
    let issuer_bytes = HostAllocationPlan::retained_view_issuer_bytes().unwrap();
    let retained = per_view + issuer_bytes;
    let ledger = ManagedHostEffectLedger::new_with_epoch(retained + per_view + 32, 11);
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let receipt = HostAllocationReceipt::admit(
        Some(&runtime),
        HostAllocationPlan::new(32, retained).unwrap(),
    )
    .unwrap()
    .unwrap();
    let permit = receipt.retained_view_permit(&runtime).unwrap();
    let claim = permit.claim_view(view).unwrap();
    let issuer = claim.issuer();
    drop(receipt);
    assert_eq!(ledger.snapshot().live_claimed_bytes, retained);
    assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
    drop(permit);
    let derived = issuer.clone().reserve_view(view).unwrap();
    assert_eq!(ledger.snapshot().live_claimed_bytes, retained + per_view);
    assert!(matches!(
        issuer.clone().reserve_view(view),
        Err(runmat_execution::resource::RetainedHostViewError::ExceedsCapacity)
    ));
    drop(claim);
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        issuer_bytes + per_view
    );
    drop(derived);
    assert_eq!(ledger.snapshot().live_claimed_bytes, issuer_bytes);
    drop(issuer);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[test]
fn finite_requires_its_managed_owner_and_cancellation_or_rebind_revokes_new_claims() {
    let plan = HostAllocationPlan::new(16, 16).unwrap();
    let unmanaged = runtime(RuntimeServicePorts::default().with_parallel(Rc::new(UnmanagedFinite)));
    assert!(matches!(
        HostAllocationReceipt::admit(Some(&unmanaged), plan),
        Err(HostAllocationError::MissingManagedOwner)
    ));
    let unbounded = runtime(RuntimeServicePorts::default());
    assert!(HostAllocationReceipt::admit(Some(&unbounded), plan)
        .unwrap()
        .is_none());
    assert!(HostAllocationReceipt::admit(None, plan).unwrap().is_none());
    assert!(HostAllocationPlan::new(u64::MAX, 1).is_err());

    // Actual sparse product ingress must use the explicit original policy even
    // when there is no ambient owner or a foreign unbounded owner is installed.
    let sparse = Value::SparseTensor(
        runmat_value::SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![2., 3.]).unwrap(),
    );
    let dense = Value::Tensor(runmat_value::Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap());
    let product = |original: &RuntimeContext| {
        futures::executor::block_on(
            crate::builtins::math::linalg::ops::matrix_arithmetic::mtimes::mtimes_eval_in_context(
                original, &sparse, &dense,
            ),
        )
    };
    assert!(crate::context::legacy::active().is_none());
    let small_ledger = ManagedHostEffectLedger::new_with_epoch(31, 3);
    let small_runtime = runtime(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(ManagedParallel(small_ledger.clone()))),
    );
    for foreign in [false, true] {
        let _ambient = foreign.then(|| unbounded.enter());
        let error = product(&small_runtime).expect_err("explicit finite sparse owner");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Placement:InsufficientResourceEvidence")
        );
    }
    assert!(matches!(
        HostAllocationReceipt::admit(Some(&small_runtime), plan),
        Err(HostAllocationError::Resource(_))
    ));
    assert_eq!(small_ledger.snapshot().live_reserved_bytes, 0);

    let ledger = ManagedHostEffectLedger::new_with_epoch(32, 3);
    let cancellation = CancellationSource::new();
    let runtime = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    )
    .with_cancellation_token(cancellation.token());
    let owner = HostAllocationReceipt::capture_owner(Some(&runtime))
        .unwrap()
        .unwrap();
    assert_eq!(
        ledger.snapshot().live_reserved_bytes,
        0,
        "a read-only checkpoint grants no bytes"
    );
    let probe = HostAllocationReceipt::admit(Some(&runtime), plan)
        .unwrap()
        .unwrap();
    drop(probe);
    owner.check(&runtime).unwrap();
    let receipt = HostAllocationReceipt::admit(Some(&runtime), plan)
        .unwrap()
        .unwrap();
    cancellation.cancel(CancellationReason::User);
    assert!(matches!(
        owner.check(&runtime),
        Err(HostAllocationError::Cancelled)
    ));
    assert!(matches!(
        receipt.checkpoint(&runtime),
        Err(HostAllocationError::Cancelled)
    ));
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    for foreign in [false, true] {
        let _ambient = foreign.then(|| unbounded.enter());
        let error = product(&runtime).expect_err("explicit cancelled sparse owner");
        assert_eq!(
            error.identifier(),
            Some(format!("{}:Cancelled", runtime.error_namespace()).as_str())
        );
        assert_eq!(
            error.failure_origin(),
            crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        let mut typed_cancelled = false;
        loop {
            typed_cancelled |= matches!(
                cause.downcast_ref::<crate::execution::ExecutionServiceError>(),
                Some(crate::execution::ExecutionServiceError::Cancelled)
            );
            let Some(next) = cause.source() else { break };
            cause = next;
        }
        assert!(
            typed_cancelled,
            "original typed cancellation cause must survive any presentation wrapper"
        );
    }

    let fresh = runtime.with_cancellation_token(CancellationSource::new().token());
    let receipt = HostAllocationReceipt::admit(Some(&fresh), plan)
        .unwrap()
        .unwrap();
    let owner = HostAllocationReceipt::capture_owner(Some(&fresh))
        .unwrap()
        .unwrap();
    {
        let other = CancellationSource::new();
        let _scope = fresh.scoped_cancellation_token(other.token());
        assert!(matches!(
            owner.check(&fresh),
            Err(HostAllocationError::StaleAuthority)
        ));
    }
    owner.check(&fresh).unwrap();
    fresh.invalidate_host_effect_session_authority();
    assert!(matches!(
        owner.check(&fresh),
        Err(HostAllocationError::StaleAuthority)
    ));
    assert!(matches!(
        receipt.checkpoint(&fresh),
        Err(HostAllocationError::StaleAuthority)
    ));
    drop(receipt);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    let owner = HostAllocationReceipt::capture_owner(Some(&fresh))
        .unwrap()
        .unwrap();
    ledger.set_epoch_for_test(4);
    assert!(matches!(
        owner.check(&fresh),
        Err(HostAllocationError::StaleAuthority)
    ));
}

#[test]
fn destination_end_rejects_rotated_owner_before_preparing_receiver() {
    use crate::sequence::SequenceDestinationBuilder;
    use futures::executor::block_on;

    let ledger = ManagedHostEffectLedger::new(65_536);
    let context = runtime(
        RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
    );
    let builder = SequenceDestinationBuilder::new_in_context(&context, &Value::Num(3.0))
        .expect("destination with original owner");
    let before = ledger.snapshot().live_reserved_bytes;
    {
        let rotated = CancellationSource::new();
        let _scope = context.scoped_cancellation_token(rotated.token());
        block_on(builder.prepare_current_receiver_for_end_in_context(&context, None))
            .expect_err("end must validate the original owner before receiver preparation");
        assert_eq!(ledger.snapshot().live_reserved_bytes, before);
    }
    block_on(builder.prepare_current_receiver_for_end_in_context(&context, None))
        .expect("original owner remains usable after the temporary scope");
    // Ordinary destinations retain finite metadata support, but capturing
    // presently unquoted method-frame backing must refuse before its factory.
    let called = std::cell::Cell::new(false);
    let frame_builder = SequenceDestinationBuilder::new_in_context(&context, &Value::Num(3.0))
        .expect("ordinary finite builder");
    let error = frame_builder
        .with_method_frame_factory_in_context(&context, || {
            called.set(true);
            Ok(crate::object::protocol::ObjectAccessContext::default())
        })
        .expect_err("method-frame backing needs a complete quote");
    assert_eq!(
        error.identifier(),
        Some("RunMat:Placement:InsufficientResourceEvidence")
    );
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert!(!called.get());
    assert_eq!(ledger.snapshot().live_reserved_bytes, before);
    context.invalidate_host_effect_session_authority();
    block_on(builder.prepare_current_receiver_for_end_in_context(&context, None))
        .expect_err("session invalidation revokes the original builder");
    drop(builder);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
