use std::rc::Rc;
mod matrix;

use runmat_value::{Tensor, Value};

use super::{execution, output, request, workspace::GmresGeometry};
use crate::context::host_effect::ManagedHostEffectLedger;
use crate::context::{
    ParallelCapability, RuntimeContext, RuntimeParallelResources, RuntimeParallelService,
    RuntimeServicePorts,
};

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

fn system() -> (Value, Value) {
    // Column-major A = [4 1; 2 3], with exact solution [0.3; -0.2].
    (
        Value::Tensor(Tensor::new(vec![4.0, 2.0, 1.0, 3.0], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gmres_public_route_preserves_solution_and_claimed_multioutput_owner() {
    let (a, b) = system();
    let parsed = parse(a, b, Vec::new()).unwrap();
    let Value::Tensor(x) =
        futures::executor::block_on(execution::execute(parsed, None, Some(1))).unwrap()
    else {
        panic!("one-output GMRES must return its native tensor");
    };
    assert!((x.as_f64_slice().unwrap()[0] - 0.3).abs() < 1e-12);
    assert!((x.as_f64_slice().unwrap()[1] + 0.2).abs() < 1e-12);

    // A nonsymmetric system with a general (not triangular) matrix inverse
    // action exercises the reusable factor/solve owner on the public route.
    let (a, b) = system();
    let empty = || Value::Tensor(Tensor::new(Vec::<f64>::new(), vec![0, 0]).unwrap());
    let preconditioner = Value::Tensor(Tensor::new(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2]).unwrap());
    let parsed = parse(a, b, vec![empty(), empty(), empty(), preconditioner]).unwrap();
    let Value::Tensor(x) =
        futures::executor::block_on(execution::execute(parsed, None, Some(1))).unwrap()
    else {
        panic!("matrix-preconditioned GMRES must return a native tensor");
    };
    assert!((x.as_f64_slice().unwrap()[0] - 0.3).abs() < 1e-12);
    assert!((x.as_f64_slice().unwrap()[1] + 0.2).abs() < 1e-12);

    let geometry = GmresGeometry::checked(2, 2, 1).unwrap();
    let plan = output::quote::<f64>(geometry, Some(5)).unwrap();
    let ledger =
        ManagedHostEffectLedger::new(plan.scratch_peak_bytes() + plan.retained_output_bytes());
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
        );
    let (a, b) = system();
    let parsed = parse(a, b, Vec::new()).unwrap();
    let Value::OutputList(outputs) =
        futures::executor::block_on(execution::execute(parsed, Some(&context), Some(5))).unwrap()
    else {
        panic!("five-output GMRES must retain its exact list owner");
    };
    assert_eq!(outputs.len(), 5);
    assert!(outputs.has_retained_claim());
    assert_eq!(outputs[1], Value::Num(0.0));
    let Value::Tensor(iter) = &outputs[3] else {
        panic!("GMRES iteration output is a tensor");
    };
    assert_eq!(iter.shape(), &[1, 2]);
    let Value::Tensor(history) = &outputs[4] else {
        panic!("GMRES residual history is a tensor");
    };
    assert!(history.as_f64_slice().unwrap().len() >= 2);
    let alias = outputs.clone();
    drop(outputs);
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(alias);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

fn parse(a: Value, b: Value, args: Vec<Value>) -> crate::BuiltinResult<request::GmresRequest> {
    request::parse(
        a,
        b,
        crate::execution::RootedValues::new(args).unwrap().into(),
    )
}
