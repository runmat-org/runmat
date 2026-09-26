use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_builtins::{
    BuiltinAsyncBehavior, BuiltinOperandPreparation, BuiltinPreparationOperation,
};
use runmat_value::{IntegerStorage, Tensor, Value};
use std::{cell::RefCell, rc::Rc};

const NAMES: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];
type Events = Rc<RefCell<Vec<&'static str>>>;
struct Observer(Events);
impl BuiltinPreparationObservation for Observer {
    fn begin_execution(&mut self) {
        self.0.borrow_mut().push("begin");
    }
    fn finish(self: Box<Self>, result: &Result<Value, RuntimeError>) -> Result<(), RuntimeError> {
        self.0
            .borrow_mut()
            .push(if result.is_ok() { "success" } else { "failure" });
        Ok(())
    }
    fn abort(self: Box<Self>) {
        self.0.borrow_mut().push("abort");
    }
}

struct Service {
    events: Events,
    replace_invalid: bool,
}
impl RuntimeAccelerationService for Service {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert!(NAMES.contains(&request.admission().binding.builtin.name));
        assert_eq!(request.form().operation, BuiltinPreparationOperation::Unary);
        assert_eq!(request.form().data_arguments, &[0]);
        assert_eq!(request.admission().signature, 0);
        self.events.borrow_mut().push("prepare");
        let operands = if self.replace_invalid {
            vec![Value::String("invalid service replacement".into())]
        } else {
            request.into_operands()
        };
        let observer = Observer(self.events.clone());
        Box::pin(async move {
            Ok(PreparedBuiltinOperands::new(
                operands,
                Some(Box::new(observer)),
            ))
        })
    }
}

fn enter(events: &Events, replace_invalid: bool) -> RuntimeContextGuard {
    RuntimeContextGuard::enter(
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_acceleration(Rc::new(
                Service {
                    events: events.clone(),
                    replace_invalid,
                },
            ))),
    )
}

#[test]
fn all_eight_contracts_declare_real_suspension_and_typed_unary_form() {
    for name in NAMES {
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        assert_eq!(
            entry.contract.async_behavior,
            BuiltinAsyncBehavior::MaySuspend
        );
        assert!(entry
            .contract
            .effects
            .contains(&runmat_types::EffectKind::MaySuspend));
        let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
            panic!("admitted form")
        };
        assert_eq!(forms.len(), 1);
        assert_eq!(forms[0].operation, BuiltinPreparationOperation::Unary);
        assert_eq!(forms[0].data_arguments, &[0]);
        assert!(runmat_builtins::validate_builtin_preparation(entry).is_empty());
    }
}

#[test]
fn invalid_calls_do_not_prepare_and_valid_shapes_observe_once() {
    let events = Events::default();
    let _context = enter(&events, false);
    for name in NAMES {
        for args in [
            vec![],
            vec![Value::String("invalid".into())],
            vec![Value::Num(1.0), Value::Num(2.0)],
        ] {
            assert!(block_on(crate::dispatcher::call_builtin_async(name, &args)).is_err());
            assert!(events.borrow().is_empty());
        }
        for (data, shape) in [
            (vec![], vec![0, 2, 3]),
            (vec![f64::NAN, 3.5], vec![1, 1, 2]),
        ] {
            let args = [Value::Tensor(Tensor::new(data, shape.clone()).unwrap())];
            let output = block_on(crate::dispatcher::call_builtin_async(name, &args)).unwrap();
            let Value::Tensor(output) = output else {
                panic!("shape-preserving typed array")
            };
            assert_eq!(output.shape, shape);
            assert!(output.integer_storage().is_some());
            assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
            events.borrow_mut().clear();
        }
    }
}

#[test]
fn wide_integer_contents_survive_preparation_without_float_materialization() {
    let events = Events::default();
    let _context = enter(&events, false);
    let args = [Value::Tensor(
        Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap(),
    )];
    let Value::Tensor(output) =
        block_on(crate::dispatcher::call_builtin_async("uint64", &args)).unwrap()
    else {
        panic!("typed output")
    };
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
    assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
}

#[test]
fn unpolled_call_has_no_service_effect_and_service_replacement_is_revalidated() {
    let events = Events::default();
    let _context = enter(&events, true);
    let args = [Value::Num(1.0)];
    drop(crate::dispatcher::call_builtin_async("int8", &args));
    assert!(events.borrow().is_empty());
    assert!(block_on(crate::dispatcher::call_builtin_async("int8", &args)).is_err());
    assert_eq!(&*events.borrow(), &["prepare", "begin", "failure"]);
}

#[test]
fn requested_output_bound_precedes_preparation_but_discard_executes() {
    let events = Events::default();
    let _context = enter(&events, false);
    for name in NAMES {
        let args = [Value::Num(3.5)];
        assert!(block_on(crate::dispatcher::call_builtin_async_with_outputs(
            name, &args, 2
        ))
        .is_err());
        assert!(events.borrow().is_empty());
        for count in [0, 1] {
            block_on(crate::dispatcher::call_builtin_async_with_outputs(
                name, &args, count,
            ))
            .unwrap();
            assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
            events.borrow_mut().clear();
        }
    }
}

#[test]
fn unsupported_complex_sparse_is_structured_error_before_preparation() {
    let events = Events::default();
    let _context = enter(&events, false);
    let value = Value::SparseTensor(
        runmat_value::SparseTensor::new_complex(2, 1, vec![0, 1], vec![1], vec![(1.0, 2.0)])
            .unwrap(),
    );
    for name in NAMES {
        let error = block_on(crate::dispatcher::call_builtin_async(
            name,
            std::slice::from_ref(&value),
        ))
        .unwrap_err();
        assert!(error.identifier().unwrap().ends_with(":InvalidInput"));
        assert!(events.borrow().is_empty());
    }
}

#[test]
fn characters_preserve_column_major_order_and_full_nd_shape() {
    for (data, shape) in [
        (vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], vec![2, 2, 2]),
        (vec![], vec![0, 2, 3]),
    ] {
        let chars = runmat_value::CharArray::new_with_shape(data, shape.clone()).unwrap();
        let expected: Vec<u16> = chars
            .to_column_major()
            .iter()
            .map(|value| *value as u16)
            .collect();
        let Value::Tensor(output) = block_on(crate::dispatcher::call_builtin_async(
            "uint16",
            &[Value::CharArray(chars)],
        ))
        .unwrap() else {
            panic!("character array conversion")
        };
        assert_eq!(output.shape, shape);
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U16(expected))
        );
    }
}
