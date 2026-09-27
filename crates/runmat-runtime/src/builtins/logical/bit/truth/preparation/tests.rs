use crate::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_builtins::LogicalBinaryOperator;
use runmat_value::{Tensor, Value};
use std::{cell::Cell, rc::Rc};

struct RecordingService(Rc<Cell<usize>>);
impl RuntimeAccelerationService for RecordingService {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert_eq!(request.form().data_arguments, &[0, 1]);
        self.0.set(self.0.get() + 1);
        Box::pin(async move {
            Ok(PreparedBuiltinOperands::passthrough(
                request.into_operands(),
            ))
        })
    }
}

#[test]
fn invalid_types_and_shapes_are_rejected_before_preparation() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(calls.clone()))),
        );
    let _context = RuntimeContextGuard::enter(context);
    let data = || Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    for op in [
        LogicalBinaryOperator::And,
        LogicalBinaryOperator::Or,
        LogicalBinaryOperator::Xor,
    ] {
        assert!(block_on(super::super::binary::execute(
            data(),
            Value::String("invalid".into()),
            op
        ))
        .is_err());
        assert!(block_on(super::super::binary::execute(
            data(),
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap()),
            op
        ))
        .is_err());
        let value = block_on(super::super::binary::execute(data(), Value::Bool(true), op)).unwrap();
        assert!(matches!(value, Value::LogicalArray(_)));
    }
    assert_eq!(calls.get(), 3);
}
