use super::*;
use crate::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use futures::executor::block_on;
use std::{cell::Cell, rc::Rc};

struct RecordingService(Rc<Cell<usize>>);
impl RuntimeAccelerationService for RecordingService {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        self.0.set(self.0.get() + 1);
        assert_eq!(
            request.admission().binding.builtin,
            LOGICAL_CATALOG_ENTRY.identity
        );
        assert_eq!(request.form().data_arguments, &[0]);
        assert!(matches!(request.operands(), [Value::Tensor(_)]));
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

fn install(calls: Rc<Cell<usize>>) -> (RuntimeContextGuard, RuntimeContext) {
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_acceleration(Rc::new(RecordingService(calls))),
        );
    (RuntimeContextGuard::enter(context.clone()), context)
}

#[test]
fn invalid_nan_arity_and_outputs_never_reach_preparation() {
    let calls = Rc::new(Cell::new(0));
    let (_selected, context) = install(calls.clone());
    let tensor = || Value::Tensor(Tensor::new(vec![0.0, 2.0], vec![1, 2]).unwrap());
    assert!(block_on(super::super::logical_builtin(
        Some(context.clone()),
        Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![])
    ))
    .is_err());
    assert!(block_on(super::super::logical_builtin(
        Some(context.clone()),
        tensor(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)])
    ))
    .is_err());
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert!(block_on(super::super::logical_builtin(
        Some(context),
        tensor(),
        crate::builtins::common::test_support::builtin_arguments(vec![])
    ))
    .is_err());
    assert_eq!(calls.get(), 0);
}

#[test]
fn admitted_dense_shape_and_zero_output_effect_are_preserved() {
    let calls = Rc::new(Cell::new(0));
    let (_selected, context) = install(calls.clone());
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let result = block_on(super::super::logical_builtin(
        Some(context),
        Value::Tensor(Tensor::new(vec![0.0, 2.0, -1.0, 0.0], vec![2, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::LogicalArray(array) = result.value() else {
        panic!("logical array")
    };
    assert_eq!(array.shape, vec![2, 2]);
    assert_eq!(array.data.as_slice(), &[0, 1, 1, 0]);
    assert_eq!(calls.get(), 1);
}
