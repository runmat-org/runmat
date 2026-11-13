use super::NAMES;
use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeServiceFuture, RuntimeServicePorts,
};
use crate::RuntimeError;
use runmat_builtins::BuiltinPreparationOperation;
use runmat_value::Value;
use std::{cell::RefCell, rc::Rc};

pub(super) type Events = Rc<RefCell<Vec<&'static str>>>;
struct Observer(Events);
impl BuiltinPreparationObservation for Observer {
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

struct Service {
    events: Events,
    replace_invalid: bool,
}
impl RuntimeAccelerationService for Service {
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
        let replace_invalid = self.replace_invalid;
        let observer = Observer(self.events.clone());
        Box::pin(async move {
            request
                .into_operands()
                .with_unbounded_values(|operands| {
                    let operands = if replace_invalid {
                        vec![Value::String("invalid service replacement".into())]
                    } else {
                        operands
                    };
                    PreparedBuiltinOperands::new(operands, Some(Box::new(observer)))
                })
                .and_then(std::convert::identity)
        })
    }
}

pub(super) fn runtime(events: &Events, replace_invalid: bool) -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_acceleration(Rc::new(Service {
                events: events.clone(),
                replace_invalid,
            })),
        )
}
