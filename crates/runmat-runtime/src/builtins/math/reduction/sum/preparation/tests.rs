use super::*;
use crate::context::preparation::{BuiltinPreparationObservation, PreparedBuiltinOperands};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_builtins::ReductionNaN;
use runmat_value::{IntValue, IntegerStorage, Tensor};
use std::{cell::RefCell, rc::Rc};

struct RecordingService(Rc<RefCell<Vec<AdmittedSumReduction>>>);

impl RuntimeAccelerationService for RecordingService {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert_eq!(request.form().data_arguments, &[0]);
        let invocation = request
            .sum_reduction()
            .expect("typed sum controls")
            .invocation;
        let entry = runmat_builtins::sum_invocation_entry(invocation);
        let runmat_builtins::BuiltinOperandPreparation::Forms([form]) = entry.placement.preparation
        else {
            panic!("sum invocation has exactly one canonical preparation form")
        };
        assert_eq!(request.admission().binding.builtin, entry.identity);
        assert!(form.signatures.contains(&request.admission().signature));
        assert!(matches!(request.admission().input_count, 1 | 2 | 4));
        assert_eq!(request.operands().len(), 1);
        self.0
            .borrow_mut()
            .push(request.sum_reduction().expect("typed sum controls").clone());
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)
                .and_then(std::convert::identity)
        })
    }
}

#[test]
fn nansum_uses_its_own_catalog_binding_with_the_shared_typed_omit_route() {
    let observed = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(observed.clone()))),
        );
    let _context = RuntimeContextGuard::enter(context.clone());
    let data = Value::Tensor(Tensor::from_f32(vec![1.0, f32::NAN, 3.0, 4.0], vec![2, 2]).unwrap());
    let output = block_on(super::super::sum_for_invocation(
        Some(&context),
        data,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        None,
        runmat_types::SumInvocation::NanSum,
    ))
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("nansum result is a tensor")
    };
    assert_eq!(output.numeric_dtype(), runmat_value::NumericDType::F32);
    assert_eq!(output.materialize_f64(), vec![1.0, 7.0]);
    let admitted = observed.borrow();
    assert_eq!(admitted.len(), 1);
    assert_eq!(admitted[0].invocation, runmat_types::SumInvocation::NanSum);
    assert_eq!(admitted[0].missing, ReductionNaN::Omit);
    assert_eq!(admitted[0].publication_class, NumericClass::Single);
    drop(admitted);
    let invalid = block_on(super::super::sum_for_invocation(
        Some(&context),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(0.0)]),
        None,
        runmat_types::SumInvocation::NanSum,
    ))
    .expect_err("invalid alias controls must fail before provider admission");
    assert_eq!(invalid.identifier(), Some("RunMat:sum:InvalidArgument"));
    assert_eq!(observed.borrow().len(), 1);
}

#[test]
fn sum_prepares_only_data_after_all_controls_are_admitted() {
    let observed = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(observed.clone()))),
        );
    let _context = RuntimeContextGuard::enter(context.clone());
    let data = || Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let result = block_on(super::super::sum_builtin(
        Some(context.clone()),
        None,
        data(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.0),
            Value::from("omitnan"),
            Value::from("double"),
        ]),
    ))
    .unwrap();
    let Value::Tensor(result) = result.value() else {
        panic!("non-scalar sum returns a tensor")
    };
    assert_eq!(result.materialize_f64(), vec![3.0, 7.0]);
    let admitted = observed.borrow();
    assert_eq!(admitted.len(), 1);
    assert_eq!(admitted[0].axes, vec![0]);
    assert_eq!(admitted[0].missing, ReductionNaN::Omit);
    assert_eq!(admitted[0].input_class, NumericClass::Single);
    assert_eq!(admitted[0].accumulation_class, NumericClass::Double);
    assert_eq!(admitted[0].publication_class, NumericClass::Double);
    drop(admitted);

    let error = block_on(super::super::sum_builtin(
        Some(context.clone()),
        None,
        data(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::from("omitnan"),
            Value::from("double"),
        ]),
    ))
    .expect_err("invalid dimension precedes preparation");
    assert_eq!(error.identifier(), Some("RunMat:sum:InvalidArgument"));
    assert_eq!(observed.borrow().len(), 1);

    let delegated = block_on(super::super::sum_for_invocation(
        Some(&context),
        data(),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        Some(ReductionNaN::Omit),
        runmat_types::SumInvocation::Sum,
    ))
    .unwrap();
    assert!(matches!(delegated, Value::Tensor(_)));
    assert_eq!(observed.borrow().len(), 2);
}

struct NativeObservation(Rc<RefCell<Vec<&'static str>>>);

impl BuiltinPreparationObservation for NativeObservation {
    fn begin_execution(&mut self) {
        self.0.borrow_mut().push("begin");
    }

    fn finish(self: Box<Self>, result: Result<&Value, &RuntimeError>) -> Result<(), RuntimeError> {
        self.0
            .borrow_mut()
            .push(if result.is_ok() { "success" } else { "failure" });
        Ok(())
    }

    fn abort(self: Box<Self>) {
        self.0.borrow_mut().push("abort");
    }
}

struct NativeService {
    controls: Rc<RefCell<Vec<AdmittedSumReduction>>>,
    events: Rc<RefCell<Vec<&'static str>>>,
}

impl RuntimeAccelerationService for NativeService {
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let controls = request
            .sum_reduction()
            .expect("native sum controls")
            .clone();
        assert_eq!(request.operands().len(), 1);
        assert_eq!(controls.accumulation_class, controls.input_class);
        assert_eq!(controls.publication_class, controls.input_class);
        self.controls.borrow_mut().push(controls);
        self.events.borrow_mut().push("prepare");
        let events = self.events.clone();
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(|values| {
                    PreparedBuiltinOperands::new(values, Some(Box::new(NativeObservation(events))))
                })
                .and_then(std::convert::identity)
        })
    }
}

#[test]
fn native_integer_sum_and_nansum_execute_inside_the_admitted_observation() {
    let controls = Rc::new(RefCell::new(Vec::new()));
    let events = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_acceleration(Rc::new(
            NativeService {
                controls: controls.clone(),
                events: events.clone(),
            },
        )));
    let _context = RuntimeContextGuard::enter(context.clone());

    let native = || {
        Value::Tensor(Tensor::new_integer(IntegerStorage::I8(vec![100, 100]), vec![2, 1]).unwrap())
    };
    for invocation in [
        runmat_types::SumInvocation::Sum,
        runmat_types::SumInvocation::NanSum,
    ] {
        let result = block_on(super::super::sum_for_invocation(
            Some(&context),
            native(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
            None,
            invocation,
        ))
        .expect("native sum");
        assert_eq!(result, Value::Int(IntValue::I8(127)));
    }
    let scalar = block_on(super::super::sum_for_invocation(
        Some(&context),
        Value::Int(IntValue::U64(u64::MAX)),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        None,
        runmat_types::SumInvocation::Sum,
    ))
    .expect("wide native scalar");
    assert_eq!(scalar, Value::Int(IntValue::U64(u64::MAX)));

    let admitted = controls.borrow();
    assert_eq!(admitted.len(), 3);
    assert_eq!(admitted[0].invocation, runmat_types::SumInvocation::Sum);
    assert_eq!(admitted[1].invocation, runmat_types::SumInvocation::NanSum);
    assert_eq!(admitted[1].missing, ReductionNaN::Omit);
    assert_eq!(admitted[0].input_class, NumericClass::Int8);
    assert_eq!(admitted[2].input_class, NumericClass::UInt64);
    drop(admitted);
    assert_eq!(
        events.borrow().as_slice(),
        [
            "prepare", "begin", "success", "prepare", "begin", "success", "prepare", "begin",
            "success"
        ]
    );

    let unpolled = super::super::sum_for_invocation(
        Some(&context),
        native(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        None,
        runmat_types::SumInvocation::Sum,
    );
    drop(unpolled);
    assert_eq!(controls.borrow().len(), 3);
    let error = block_on(super::super::sum_for_invocation(
        Some(&context),
        native(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(0.0),
            Value::from("native"),
        ]),
        None,
        runmat_types::SumInvocation::Sum,
    ))
    .expect_err("invalid dimension must precede preparation");
    assert_eq!(error.identifier(), Some("RunMat:sum:InvalidArgument"));
    assert_eq!(controls.borrow().len(), 3);
}
