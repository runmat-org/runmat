use crate::context::preparation::{
    AdmittedBuiltinPreparation, BuiltinPreparationObservation, PreparedBuiltinOperands,
};
use crate::context::{
    RuntimeAccelerationService, RuntimeContext, RuntimeContextGuard, RuntimeServiceFuture,
    RuntimeServicePorts,
};
use crate::{dispatcher, RuntimeError};
use futures::executor::block_on;
use runmat_builtins::{
    BuiltinOperandPreparation, BuiltinPreparationOperation, BuiltinSinkPreparation,
};
use runmat_value::{NumericDType, Tensor, Value};
use std::{
    cell::RefCell,
    future::Future,
    rc::Rc,
    task::{Context, Poll},
};

const NAMES: [&str; 3] = ["gray2rgb", "rgb2gray", "rgb2hsv"];
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
    pause: bool,
}
impl RuntimeAccelerationService for Service {
    fn supports_operation(&self, operation: &str) -> bool {
        NAMES.contains(&operation)
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        assert!(NAMES.contains(&request.admission().binding.builtin.name));
        assert_eq!(
            request.form().operation,
            BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather)
        );
        assert_eq!(request.form().data_arguments, &[0]);
        self.events.borrow_mut().push("prepare");
        let prepared = PreparedBuiltinOperands::new(
            request.into_operands(),
            Some(Box::new(Observer(self.events.clone()))),
        );
        let pause = self.pause;
        Box::pin(async move {
            if pause {
                std::future::pending::<()>().await;
            }
            Ok(prepared)
        })
    }
}

fn enter(events: &Events, pause: bool) -> RuntimeContextGuard {
    RuntimeContextGuard::enter(
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_acceleration(Rc::new(
                Service {
                    events: events.clone(),
                    pause,
                },
            ))),
    )
}

fn valid(name: &str) -> Value {
    if name == "gray2rgb" {
        Value::Tensor(Tensor::new_with_dtype(vec![0.0, 1.0], vec![1, 2], NumericDType::U8).unwrap())
    } else {
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 1, 3]).unwrap())
    }
}

#[test]
fn color_forms_are_typed_gather_and_valid_calls_observe_once() {
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let events = Events::default();
    let _context = enter(&events, false);
    for name in NAMES {
        let entry = runmat_builtins::builtin_catalog_primary_entry_by_name(name).unwrap();
        let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
            panic!("missing form");
        };
        assert_eq!(forms.len(), 1);
        assert_eq!(forms[0].signatures, &[0]);
        assert_eq!(forms[0].data_arguments, &[0]);
        assert_eq!(
            forms[0].operation,
            BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather)
        );
        assert!(runmat_builtins::validate_builtin_preparation(entry).is_empty());
        block_on(dispatcher::call_builtin_async(name, &[valid(name)])).unwrap();
        assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
        events.borrow_mut().clear();
    }
}

#[test]
fn known_invalid_geometry_class_and_output_count_do_not_prepare() {
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let events = Events::default();
    let _context = enter(&events, false);
    for name in NAMES {
        assert!(block_on(dispatcher::call_builtin_async_with_outputs(
            name,
            &[valid(name)],
            2
        ))
        .is_err());
        assert!(events.borrow().is_empty());
        let invalid = if name == "gray2rgb" {
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 1, 3]).unwrap())
        } else {
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap())
        };
        // Nx3 is valid only as a double colormap; uint8 must reject.
        let invalid = if name == "gray2rgb" {
            invalid
        } else {
            Value::Tensor(
                Tensor::new_with_dtype(vec![1.0; 3], vec![1, 3], NumericDType::U8).unwrap(),
            )
        };
        assert!(block_on(dispatcher::call_builtin_async(name, &[invalid])).is_err());
        assert!(events.borrow().is_empty());
    }
}

#[test]
fn unsupported_resident_metadata_rejects_before_service_or_download() {
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let events = Events::default();
    let _context = enter(&events, false);
    for name in NAMES {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        let (shape, dtype) = if name == "gray2rgb" {
            (
                vec![1, 1, 3],
                runmat_accelerate_api::NumericElementType::F64,
            )
        } else {
            (
                vec![1, 1, 3],
                runmat_accelerate_api::NumericElementType::I16,
            )
        };
        let handle = owner
            .allocate_zeroed(shape, dtype, runmat_accelerate_api::GpuTensorStorage::Real)
            .unwrap()
            .handle;
        let error = block_on(dispatcher::call_builtin_async(
            name,
            &[Value::GpuTensor(handle)],
        ))
        .unwrap_err();
        let expected = format!("RunMat:{name}:InvalidInput");
        assert_eq!(error.identifier(), Some(expected.as_str()));
        assert!(events.borrow().is_empty());
    }
}

#[test]
fn dropped_unpolled_and_polled_calls_keep_observation_transactional() {
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    let events = Events::default();
    let _context = enter(&events, true);
    for name in NAMES {
        let args = [valid(name)];
        drop(dispatcher::call_builtin_async(name, &args));
        assert!(events.borrow().is_empty());
        let mut call = Box::pin(dispatcher::call_builtin_async(name, &args));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(&*events.borrow(), &["prepare"]);
        drop(call);
        assert_eq!(&*events.borrow(), &["prepare", "abort"]);
        events.borrow_mut().clear();
    }
}
