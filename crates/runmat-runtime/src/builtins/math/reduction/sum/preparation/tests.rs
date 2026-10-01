use super::*;
use crate::context::preparation::PreparedBuiltinOperands;
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_builtins::ReductionNaN;
use runmat_value::Tensor;
use std::{cell::RefCell, rc::Rc};

struct RecordingService(Rc<RefCell<Vec<AdmittedSumReduction>>>);

impl RuntimeAccelerationService for RecordingService {
    fn supports_operation(&self, _: &str) -> bool {
        true
    }

    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert_eq!(request.form().data_arguments, &[0]);
        assert_eq!(request.admission().signature, 8);
        assert_eq!(request.admission().input_count, 4);
        assert_eq!(request.operands().len(), 1);
        self.0
            .borrow_mut()
            .push(request.sum_reduction().expect("typed sum controls").clone());
        Box::pin(async move {
            Ok(PreparedBuiltinOperands::passthrough(
                request.into_operands(),
            ))
        })
    }
}

#[test]
fn sum_prepares_only_data_after_all_controls_are_admitted() {
    let observed = Rc::new(RefCell::new(Vec::new()));
    let context = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default()
                .with_acceleration(Rc::new(RecordingService(observed.clone()))),
        );
    let _context = RuntimeContextGuard::enter(context);
    let data = || Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let result = block_on(super::super::sum_builtin(
        data(),
        vec![
            Value::Num(1.0),
            Value::from("omitnan"),
            Value::from("double"),
        ],
    ))
    .unwrap();
    let Value::Tensor(result) = result else {
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
        data(),
        vec![
            Value::Num(0.0),
            Value::from("omitnan"),
            Value::from("double"),
        ],
    ))
    .expect_err("invalid dimension precedes preparation");
    assert_eq!(error.identifier(), Some("RunMat:sum:InvalidArgument"));
    assert_eq!(observed.borrow().len(), 1);

    let delegated = block_on(super::super::sum_with_nan_policy(
        data(),
        Vec::new(),
        Some(ReductionNaN::Omit),
    ))
    .unwrap();
    assert!(matches!(delegated, Value::Tensor(_)));
    assert_eq!(observed.borrow().len(), 1);
}
