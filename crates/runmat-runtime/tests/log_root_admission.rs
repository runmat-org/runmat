use futures::executor::block_on;
use runmat_accelerate_api::NumericalDomainEvidence;
use runmat_runtime::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use runmat_runtime::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use runmat_runtime::RuntimeError;
use runmat_types::{RealUnaryClassification, RealUnaryPredicate};
use runmat_value::{IntValue, Tensor, Value};
use std::{cell::RefCell, rc::Rc};

struct Recorder(Rc<RefCell<Vec<(&'static str, RealUnaryPredicate)>>>);
impl RuntimeAccelerationService for Recorder {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let NumericalDomainEvidence::RealUnary {
            predicate,
            classification,
        } = request.numerical_domain()
        else {
            panic!("typed evidence");
        };
        assert_eq!(classification, RealUnaryClassification::Inside);
        assert_eq!(request.form().data_arguments, &[0]);
        assert_eq!(request.admission().signature, 0);
        self.0
            .borrow_mut()
            .push((request.admission().binding.builtin.name, predicate));
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

fn tensor(values: Vec<f64>) -> Value {
    let length = values.len();
    Value::Tensor(Tensor::new(values, vec![1, length]).unwrap())
}

#[test]
fn actual_dispatch_admits_only_valid_real_domain_forms_before_preparation() {
    let records = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        RuntimeServicePorts::default().with_acceleration(Rc::new(Recorder(records.clone()))),
    );
    for name in ["log", "log10", "log1p", "log2", "sqrt", "realsqrt"] {
        let predicate = if name == "log1p" {
            RealUnaryPredicate::AtLeastMinusOne
        } else {
            RealUnaryPredicate::Nonnegative
        };
        for outputs in [0, 1] {
            let before = records.borrow().len();
            block_on(
                context.scope(runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context,
                    name,
                    &[tensor(vec![1.0, 4.0])],
                    outputs,
                )),
            )
            .unwrap();
            assert_eq!(records.borrow().len(), before + 1);
            assert_eq!(records.borrow().last(), Some(&(name, predicate)));
        }
        let before = records.borrow().len();
        for args in [
            vec![],
            vec![Value::String("invalid".into())],
            vec![tensor(vec![1.0]), Value::Num(2.0)],
        ] {
            assert!(block_on(context.scope(
                runmat_runtime::call_builtin_async_with_outputs_in_context(
                    &context, name, &args, 1
                )
            ))
            .is_err());
        }
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[tensor(vec![1.0])],
                3
            )
        ))
        .is_err());
        let negative = block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[tensor(vec![-4.0, 1.0])],
                1,
            ),
        ));
        if name == "realsqrt" {
            assert!(negative.is_err());
        } else {
            assert!(matches!(negative.unwrap().value(), Value::ComplexTensor(_)));
        }
        block_on(
            context.scope(runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[tensor(vec![f64::NAN])],
                1,
            )),
        )
        .unwrap();
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        assert!(block_on(context.scope(
            runmat_runtime::call_builtin_async_with_outputs_in_context(
                &context,
                name,
                &[Value::Int(IntValue::U16(4))],
                1
            )
        ))
        .is_err());
        assert_eq!(
            records.borrow().len(),
            before,
            "{name}: no preparation for rejected or non-real-domain forms"
        );
    }
    let before = records.borrow().len();
    block_on(
        context.scope(runmat_runtime::call_builtin_async_with_outputs_in_context(
            &context,
            "log2",
            &[tensor(vec![1.0, 4.0])],
            2,
        )),
    )
    .unwrap();
    assert_eq!(
        records.borrow().len(),
        before,
        "host-only dissection is not the one-output provider form"
    );
}
