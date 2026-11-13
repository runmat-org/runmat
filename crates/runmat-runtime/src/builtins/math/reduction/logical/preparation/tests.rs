use super::*;
use crate::context::preparation::{AdmittedBuiltinPreparation, PreparedBuiltinOperands};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::RuntimeError;
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
        assert_eq!(request.form().data_arguments, &[0]);
        assert_eq!(request.admission().signature, 6);
        assert_eq!(request.admission().input_count, 4);
        assert_eq!(request.operands().len(), 1);
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

#[test]
fn variadic_controls_are_parsed_once_and_only_data_reaches_service() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(calls.clone()))),
        );
    let data = || Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]).unwrap());
    let empty = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    for config in [
        &super::super::super::all::CONFIG,
        &super::super::super::any::CONFIG,
    ] {
        let result = block_on(super::super::execute(
            Some(&context),
            config,
            data(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                empty(),
                Value::Num(2.0),
                empty(),
            ]),
        ))
        .unwrap();
        let Value::LogicalArray(array) = result.value() else {
            panic!("logical array")
        };
        assert_eq!(array.shape, vec![2, 1]);
        assert!(block_on(super::super::execute(
            Some(&context),
            config,
            data(),
            crate::builtins::common::test_support::builtin_arguments(vec![
                empty(),
                Value::Num(0.0),
                empty()
            ])
        ))
        .is_err());
    }
    // all is provider-eligible; default any retains its existing exact host
    // route without first uploading the dense operand.
    assert_eq!(calls.get(), 1);
}

#[test]
fn exact_any_route_skips_preparation_for_default_and_explicit_omit_nan() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(calls.clone()))),
        );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let config = &super::super::super::any::CONFIG;
    for options in [vec![], vec![Value::String("omitnan".into())]] {
        for (tensor, expected) in [
            (Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap(), false),
            (Tensor::new(vec![], vec![0, 1]).unwrap(), false),
            (
                Tensor::new_integer(
                    runmat_value::IntegerStorage::U64(vec![u64::MAX]),
                    vec![1, 1],
                )
                .unwrap(),
                true,
            ),
        ] {
            let result = block_on(super::super::execute(
                Some(&context),
                config,
                Value::Tensor(tensor),
                crate::builtins::common::test_support::builtin_arguments(options.clone()),
            ))
            .unwrap();
            assert!(matches!(result.value(), Value::Bool(value) if *value == expected));
        }
    }
    assert_eq!(calls.get(), 0);
}

#[test]
fn include_nan_any_remains_provider_eligible_after_option_admission() {
    let calls = Rc::new(Cell::new(0));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(calls.clone()))),
        );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let empty = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    let result = block_on(super::super::execute(
        Some(&context),
        &super::super::super::any::CONFIG,
        Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            empty(),
            Value::String("includenan".into()),
            empty(),
        ]),
    ))
    .unwrap();
    assert!(matches!(result.value(), Value::Bool(true)));
    assert_eq!(calls.get(), 1);
}

#[test]
fn absent_explicit_context_is_passthrough_even_with_ambient_service() {
    let calls = Rc::new(Cell::new(0));
    let ambient = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(calls.clone()))),
        );
    let _ambient = RuntimeContextGuard::enter(ambient);
    let value = Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![1, 2]).unwrap());
    let result = block_on(super::super::execute(
        None,
        &super::super::super::all::CONFIG,
        value,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("passthrough reduction");
    assert!(matches!(result.value(), Value::Bool(false)));
    assert_eq!(calls.get(), 0);
}
