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
fn invalid_types_and_shapes_are_rejected_before_preparation() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(calls.clone()))),
        );
    let _context = RuntimeContextGuard::enter(context.clone());
    let data = || Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    for op in [
        LogicalBinaryOperator::And,
        LogicalBinaryOperator::Or,
        LogicalBinaryOperator::Xor,
    ] {
        assert!(block_on(super::super::binary::execute(
            Some(&context),
            data(),
            Value::String("invalid".into()),
            op
        ))
        .is_err());
        assert!(block_on(super::super::binary::execute(
            Some(&context),
            data(),
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap()),
            op
        ))
        .is_err());
        let value = block_on(super::super::binary::execute(
            Some(&context),
            data(),
            Value::Bool(true),
            op,
        ))
        .unwrap();
        assert!(matches!(value.value(), Value::LogicalArray(_)));
    }
    assert_eq!(calls.get(), 3);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rank_zero_and_rank_one_complex_controls_preserve_logical_shapes() {
    use runmat_value::ComplexTensor;

    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for shape in [Vec::new(), vec![1]] {
        let control =
            || Value::ComplexTensor(ComplexTensor::new(vec![(0.0, 1.0)], shape.clone()).unwrap());
        let result = block_on(super::super::binary::execute(
            None,
            Value::Tensor(Tensor::new(vec![0.0, 2.0], vec![2]).unwrap()),
            control(),
            LogicalBinaryOperator::And,
        ))
        .unwrap();
        let Value::LogicalArray(output) = result.value() else {
            panic!("expected logical array");
        };
        assert_eq!(output.shape, vec![2]);
        assert_eq!(output.data, vec![0, 1]);
        let scalar = block_on(super::super::binary::execute(
            None,
            Value::Tensor(Tensor::new(vec![2.0], shape.clone()).unwrap()),
            control(),
            LogicalBinaryOperator::And,
        ))
        .unwrap();
        assert_eq!(scalar.value(), &Value::Bool(true));
    }
}
