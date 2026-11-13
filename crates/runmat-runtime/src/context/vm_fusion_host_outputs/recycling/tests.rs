use super::*;
use crate::builtins::common::spec::{FusionHostOutputDomain, FusionHostResourceBound};
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::preparation::{AdmittedHostResources, PreparedHostNumericalOutput};
use crate::context::{
    ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
};
use runmat_builtins::BuiltinPreparationOperation;
use runmat_types::{NumericClass, NumericDomain, OperatorKind};
use std::rc::Rc;

struct ManagedParallel(ManagedHostEffectLedger);

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

fn work(
    runtime: &RuntimeContext,
    dtype: NumericDType,
    shape: &[usize],
    operation: BuiltinPreparationOperation,
) -> PreparedHostNumericalOutput {
    let operand_shapes = match operation {
        BuiltinPreparationOperation::ElementwiseBinary => {
            vec![vec![1, 1], shape.to_vec()]
        }
        _ => vec![shape.to_vec()],
    };
    let prepared_scratch_bytes =
        crate::builtins::math::host_resources::prepared_scratch_for_shapes(
            operation,
            &operand_shapes,
            shape,
        )
        .unwrap();
    let resources = AdmittedHostResources {
        bound: FusionHostResourceBound {
            retained_bytes: HostAllocationPlan::retained_tensor_bytes(dtype, shape).unwrap(),
            scratch_bytes: prepared_scratch_bytes,
        },
        prepared_scratch_bytes,
        output_shape: shape.to_vec(),
        output_class: NumericClass::Single,
        output_domain: FusionHostOutputDomain::Exact(NumericDomain::Real),
        operand_metadata: 0,
        operand_shapes,
        identity: runmat_execution::Digest::from_bytes([0; 32]),
        operation,
    };
    PreparedHostNumericalOutput::prepare_recycling_work(runtime, &resources, dtype, shape).unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn whole_tensor_slot_recycles_only_after_last_read_and_refunds_on_final_drop() {
    let ledger = ManagedHostEffectLedger::new(1 << 20);
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    let shape = [1, 2];
    let dtype = NumericDType::F32;
    let specs = [
        VmFusionHostRealSlotSpec {
            dtype,
            shape: &shape,
        },
        VmFusionHostRealSlotSpec {
            dtype,
            shape: &shape,
        },
    ];
    let neg = VmFusionHostInstruction::Intrinsic(OperatorKind::UnaryMinus);
    let sin = VmFusionHostInstruction::ElementwiseCall;
    let sin_entry = runmat_builtins::builtin_catalog_entry_by_name("sin").unwrap();
    let sin_binding = sin_entry.binding_identity(&sin_entry.bindings[0]);
    let plus = VmFusionHostInstruction::Intrinsic(OperatorKind::UnaryPlus);
    let mul = VmFusionHostInstruction::Intrinsic(OperatorKind::MatrixMultiply);
    let mut outputs =
        VmFusionHostOutputs::prepare_recycled_with(&runtime, &specs, 4, |index, _| {
            if index == 1 {
                return Ok(VmFusionHostOutputEntry::recycled_builtin(
                    sin_binding,
                    sin,
                    1,
                    [
                        Some(VmFusionHostRetirement {
                            operand_position: 0,
                            slot: 0,
                        }),
                        None,
                    ],
                    work(&runtime, dtype, &shape, BuiltinPreparationOperation::Unary),
                ));
            }
            let (operator, instruction, slot, retirement, operation) = match index {
                0 => (
                    OperatorKind::UnaryMinus,
                    neg,
                    Some(0),
                    [None, None],
                    Some(BuiltinPreparationOperation::Unary),
                ),
                2 => (OperatorKind::UnaryPlus, plus, None, [None, None], None),
                3 => (
                    OperatorKind::MatrixMultiply,
                    mul,
                    Some(0),
                    [
                        Some(VmFusionHostRetirement {
                            operand_position: 0,
                            slot: 1,
                        }),
                        None,
                    ],
                    Some(BuiltinPreparationOperation::ElementwiseBinary),
                ),
                _ => unreachable!(),
            };
            Ok(VmFusionHostOutputEntry::recycled_intrinsic(
                operator,
                instruction,
                slot,
                retirement,
                operation.map(|operation| work(&runtime, dtype, &shape, operation)),
            ))
        })
        .unwrap();

    let first = outputs
        .take_intrinsic(&runtime, neg, OperatorKind::UnaryMinus)
        .unwrap()
        .generate_real_f32(&runtime, |index| Ok((index + 1) as f32))
        .unwrap();
    let Value::Tensor(first) = first else {
        panic!("first Tensor")
    };
    let first_ptr = first.as_f32_slice().unwrap().as_ptr();
    outputs
        .retire_numerical_operands(neg, [None, None])
        .unwrap();

    let second = outputs
        .take_builtin(&runtime, sin, sin_binding)
        .unwrap()
        .generate_real_f32(&runtime, |index| {
            Ok(first.as_f32_slice().unwrap()[index].sin())
        })
        .unwrap();
    let Value::Tensor(second) = second else {
        panic!("second Tensor")
    };
    assert_ne!(first_ptr, second.as_f32_slice().unwrap().as_ptr());
    outputs
        .retire_builtin_operands(sin, vec![Value::Tensor(first)])
        .unwrap();
    outputs
        .retire_numerical_operands(plus, [None, None])
        .unwrap();

    let final_value = outputs
        .take_intrinsic(&runtime, mul, OperatorKind::MatrixMultiply)
        .unwrap()
        .generate_real_f32(&runtime, |index| {
            Ok(second.as_f32_slice().unwrap()[index] * 2.0)
        })
        .unwrap();
    let Value::Tensor(final_tensor) = final_value else {
        panic!("final Tensor")
    };
    assert_eq!(final_tensor.as_f32_slice().unwrap().as_ptr(), first_ptr);
    assert_eq!(
        final_tensor.as_f32_slice().unwrap(),
        &[1.0f32.sin() * 2.0, 2.0f32.sin() * 2.0]
    );
    outputs
        .retire_numerical_operands(mul, [Some(Value::Tensor(second)), None])
        .unwrap();
    assert!(outputs.complete());
    drop(outputs);
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        HostAllocationPlan::retained_tensor_bytes(dtype, &shape).unwrap()
    );
    drop(final_tensor);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}
