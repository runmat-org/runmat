use super::*;
use crate::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use futures::executor::block_on;
use runmat_builtins::BuiltinPreparationOperation;
use std::{cell::RefCell, rc::Rc};

type Calls = Rc<RefCell<Vec<(usize, Option<usize>)>>>;
struct Service(Calls);
impl RuntimeAccelerationService for Service {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let count = request.operands().len();
        assert_eq!(
            request.form().operation,
            BuiltinPreparationOperation::ElementwiseNary { data_arity: count }
        );
        assert_eq!(request.admission().signature, count - 2);
        assert_eq!(
            request.form().data_arguments,
            if count == 2 {
                &[0, 1][..]
            } else {
                &[0, 1, 2][..]
            }
        );
        self.0
            .borrow_mut()
            .push((count, request.admission().requested_outputs));
        let result = request
            .into_operands()
            .with_unbounded_values(PreparedBuiltinOperands::passthrough)
            .and_then(std::convert::identity);
        Box::pin(async move { result })
    }
}

#[test]
fn pol2cart_preparation_preserves_every_operand_output_mode_and_full_broadcast() {
    let calls = Calls::default();
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_acceleration(Rc::new(Service(calls.clone()))),
        );
    let all = [
        Value::Tensor(Tensor::new(vec![0.0; 2], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![2.0; 3], vec![1, 3]).unwrap()),
        Value::Tensor(Tensor::new(vec![7.0; 4], vec![1, 1, 4]).unwrap()),
    ];
    for count in 2..=3 {
        for outputs in 0..=count {
            let expected = block_on(crate::dispatcher::call_builtin_async_with_outputs(
                NAME,
                &all[..count],
                outputs,
            ))
            .unwrap();
            let actual = block_on(
                crate::dispatcher::call_builtin_async_with_outputs_in_context(
                    &context,
                    NAME,
                    &all[..count],
                    outputs,
                ),
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(calls.borrow_mut().pop(), Some((count, Some(outputs))));
        }
        block_on(crate::dispatcher::call_builtin_async_in_context(
            &context,
            NAME,
            &all[..count],
        ))
        .unwrap();
        assert_eq!(calls.borrow_mut().pop(), Some((count, None)));
    }
    assert!(block_on(
        crate::dispatcher::call_builtin_async_with_outputs_in_context(&context, NAME, &all[..2], 3)
    )
    .is_err());
    let mut invalid = all.to_vec();
    invalid[2] = Value::String("invalid elevation".into());
    assert!(block_on(crate::dispatcher::call_builtin_async_in_context(
        &context, NAME, &invalid
    ))
    .is_err());
    let mut mixed = all.to_vec();
    mixed[2] = Value::Tensor(Tensor::from_f32(vec![7.0; 4], vec![1, 1, 4]).unwrap());
    block_on(crate::dispatcher::call_builtin_async_in_context(
        &context, NAME, &mixed,
    ))
    .unwrap();
    assert!(
        calls.borrow().is_empty(),
        "invalid or mixed-class form must not be promoted"
    );
}
