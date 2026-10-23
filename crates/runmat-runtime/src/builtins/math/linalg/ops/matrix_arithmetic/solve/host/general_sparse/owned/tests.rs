use crate::builtins::math::linalg::ops::matrix_arithmetic::solve::{
    evaluate_with_context, SolveOrientation,
};
use crate::context::host_allocation::{HostAllocationError, HostAllocationPlan};
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};
use crate::execution::{
    ExecutionServiceError, OwnedCpuJob, RuntimeBlockingCpuService, RuntimeExecutionService,
};
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_value::{NumericDType, SparseTensor, Tensor, Value};
use std::{cell::Cell, rc::Rc};

struct Parallel(ManagedHostEffectLedger);
impl RuntimeParallelService for Parallel {
    fn supports(&self, _: ParallelCapability) -> bool {
        false
    }
    fn placement_resources(&self) -> RuntimeParallelResources {
        let state = self.0.snapshot();
        RuntimeParallelResources {
            memory_available_bytes: Some(state.free_bytes()),
            epoch: state.epoch,
            ..Default::default()
        }
    }
    fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
        Some(self.0.clone())
    }
}
#[derive(Default)]
struct InlineCpu(Cell<usize>);
impl RuntimeBlockingCpuService for InlineCpu {
    fn submit(
        &self,
        runtime: Option<&RuntimeContext>,
        job: OwnedCpuJob,
    ) -> crate::execution::OwnedCpuSubmission {
        let submitted = crate::execution::admit_optional_hostcall_with(
            runtime,
            ExecutionServiceError::HostAllocation,
            || async { Ok(()) },
        )?;
        self.0.set(self.0.get() + 1);
        job.run();
        Ok(submitted)
    }
}
fn context(ledger: &ManagedHostEffectLedger, service: Rc<InlineCpu>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
        RuntimeServicePorts::default()
            .with_parallel(Rc::new(Parallel(ledger.clone())))
            .with_blocking_cpu(service),
    )
}
fn solve(lhs: &Value, rhs: &Value, context: &RuntimeContext) -> BuiltinResult<Value> {
    block_on(evaluate_with_context(
        SolveOrientation::Left,
        lhs,
        rhs,
        Some(context),
    ))
}
fn empty(ledger: &ManagedHostEffectLedger) {
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

mod cases;
use cases::Class;

#[test]
fn direct_square_native_classes_retain_exact_output_and_shape_through_clone() {
    for class in [
        Class::RealF32,
        Class::RealF64,
        Class::ComplexF32,
        Class::ComplexF64,
    ] {
        let ledger = ManagedHostEffectLedger::new(2_000_000);
        let cpu = Rc::new(InlineCpu::default());
        let context = context(&ledger, cpu.clone());
        let (lhs, rhs) = class.inputs();
        let output = solve(&lhs, &rhs, &context).unwrap();
        class.verify(&output);
        assert_eq!(
            cpu.0.get(),
            1,
            "one actual ordering/factorization, no replay"
        );
        assert_eq!(ledger.snapshot().live_claimed_bytes, class.quote());
        assert_eq!(ledger.snapshot().live_reserved_bytes, class.quote());
        let clone = output.clone();
        drop(output);
        class.verify(&clone);
        assert_eq!(ledger.snapshot().live_claimed_bytes, class.quote());
        drop(clone);
        empty(&ledger);
    }
}

#[test]
fn ordinary_unbounded_general_sparse_path_preserves_native_results() {
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    for class in [
        Class::RealF32,
        Class::RealF64,
        Class::ComplexF32,
        Class::ComplexF64,
    ] {
        let (lhs, rhs) = class.inputs();
        class.verify(&solve(&lhs, &rhs, &context).unwrap());
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn output_capacity_denial_after_factor_is_terminal_and_releases_every_claim() {
    let ledger = ManagedHostEffectLedger::new(128_000);
    let cpu = Rc::new(InlineCpu::default());
    let context = context(&ledger, cpu.clone());
    let (lhs, _) = Class::RealF64.inputs();
    // Small factor fits; the private 2-by-10000 native result cannot fit.
    // RHS is borrowed: allocating another RHS payload would be a regression.
    let rhs = Value::Tensor(Tensor::new(vec![1.0; 20_000], vec![2, 10_000]).unwrap());
    assert!(
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[2, 10_000]).unwrap()
            > 128_000
    );
    let error = solve(&lhs, &rhs, &context).unwrap_err();
    assert_eq!(cpu.0.get(), 1, "factor ran once before the result denial");
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let mut cursor: &dyn std::error::Error = &error;
    let mut typed = false;
    loop {
        if let Some(cause) = cursor.downcast_ref::<HostAllocationError>() {
            typed = matches!(cause, HostAllocationError::Resource(_));
            break;
        }
        let Some(source) = cursor.source() else {
            break;
        };
        cursor = source;
    }
    assert!(typed, "allocation cause must survive the builtin mapping");
    empty(&ledger);
}

#[test]
fn singular_general_sparse_selects_minimum_norm_once() {
    let ledger = ManagedHostEffectLedger::new(2_000_000);
    let cpu = Rc::new(InlineCpu::default());
    let context = context(&ledger, cpu.clone());
    let lhs = Value::SparseTensor(
        SparseTensor::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![1.0; 4]).unwrap(),
    );
    let rhs = Value::Tensor(Tensor::new(vec![1.0; 2], vec![2, 1]).unwrap());
    let output = solve(&lhs, &rhs, &context).unwrap();
    let Value::Tensor(tensor) = &output else {
        panic!("real result required")
    };
    assert_eq!(tensor.shape(), &[2, 1]);
    for &value in tensor.as_f64_slice().unwrap() {
        assert!((value - 0.5).abs() < 1e-12);
    }
    assert_eq!(cpu.0.get(), 1);
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[2, 1]).unwrap()
    );
    drop(output);
    empty(&ledger);
}

mod complete_orthogonal;
