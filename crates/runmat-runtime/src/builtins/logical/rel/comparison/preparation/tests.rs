use super::*;
use crate::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_value::Tensor;
use std::{cell::Cell, rc::Rc};

struct RecordingService(Rc<Cell<usize>>);
impl RuntimeAccelerationService for RecordingService {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert_eq!(request.form().data_arguments, &[0, 1]);
        self.0.set(self.0.get() + 1);
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

#[test]
fn relational_numeric_admission_precedes_service_and_preserves_broadcast() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(calls.clone()))),
        );
    let _context = RuntimeContextGuard::enter(context.clone());
    let data = || Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    for op in [
        RelationalOperator::Equal,
        RelationalOperator::NotEqual,
        RelationalOperator::LessThan,
        RelationalOperator::LessThanOrEqual,
        RelationalOperator::GreaterThan,
        RelationalOperator::GreaterThanOrEqual,
    ] {
        assert!(block_on(super::super::evaluate(
            Some(&context),
            data(),
            Value::String("invalid".into()),
            op
        ))
        .is_err());
        assert!(block_on(super::super::evaluate(
            Some(&context),
            data(),
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap()),
            op
        ))
        .is_err());
        let result = block_on(super::super::evaluate(
            Some(&context),
            data(),
            Value::Num(0.0),
            op,
        ))
        .unwrap();
        let Value::LogicalArray(array) = result.value() else {
            panic!("logical array")
        };
        assert_eq!(array.shape, vec![1, 2]);
    }
    assert_eq!(calls.get(), 6);
}
