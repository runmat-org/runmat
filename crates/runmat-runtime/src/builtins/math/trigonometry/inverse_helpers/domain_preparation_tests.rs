use crate::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
use runmat_value::{Tensor, Value};
use std::{cell::RefCell, rc::Rc};

struct Service(Rc<RefCell<Vec<NumericalDomainEvidence>>>);
impl RuntimeAccelerationService for Service {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        self.0.borrow_mut().push(request.numerical_domain());
        assert_eq!(request.admission().binding.variant, "default");
        assert_eq!(request.form().data_arguments, &[0]);
        let output = request
            .into_operands()
            .with_unbounded_values(PreparedBuiltinOperands::passthrough)
            .and_then(std::convert::identity);
        Box::pin(async move { output })
    }
}

#[test]
fn inverse_real_domain_admission_precedes_service_and_preserves_exact_values() {
    let events = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_acceleration(Rc::new(Service(events.clone()))),
        );
    for name in ["asin", "acos", "acosh", "atanh"] {
        let inside = if name == "acosh" { 2.0 } else { 0.5 };
        let outside = if name == "acosh" { 0.5 } else { 2.0 };
        let value = Value::Tensor(Tensor::from_f32(vec![inside], vec![1, 1]).unwrap());
        let expected = block_on(crate::dispatcher::call_builtin_async_with_outputs(
            name,
            std::slice::from_ref(&value),
            1,
        ))
        .unwrap();
        for outputs in [0, 1] {
            let output = block_on(
                crate::dispatcher::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    std::slice::from_ref(&value),
                    outputs,
                ),
            )
            .unwrap();
            if outputs == 1 {
                assert_eq!(output, expected);
            }
            assert_eq!(
                events.borrow_mut().pop(),
                Some(NumericalDomainEvidence::RealUnary {
                    predicate: if name == "acosh" {
                        RealUnaryPredicate::AtLeastOne
                    } else {
                        RealUnaryPredicate::ClosedUnitInterval
                    },
                    classification: RealUnaryClassification::Inside,
                })
            );
        }
        assert!(block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[value],
                2
            )
        )
        .is_err());
        assert!(block_on(
            crate::dispatcher::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[Value::String("bad".into())],
                1
            )
        )
        .is_err());
        for data in [outside, f32::NAN] {
            let value = Value::Tensor(Tensor::from_f32(vec![data], vec![1, 1]).unwrap());
            block_on(
                crate::dispatcher::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    &[value],
                    1,
                ),
            )
            .unwrap();
        }
        assert!(
            events.borrow().is_empty(),
            "{name}: invalid or unproved inputs reached service"
        );
    }
}
