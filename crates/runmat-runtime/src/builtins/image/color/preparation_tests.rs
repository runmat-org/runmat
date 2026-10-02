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

const NAMES: [&str; 4] = ["gray2rgb", "hsv2rgb", "rgb2gray", "rgb2hsv"];
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
    replacement: Option<Value>,
}
impl RuntimeAccelerationService for Service {
    fn supports_operation(&self, operation: &str) -> bool {
        NAMES.contains(&operation) || operation == "ind2rgb"
    }
    fn prepare_admitted(
        &self,
        _: RuntimeContext,
        request: AdmittedBuiltinPreparation,
    ) -> RuntimeServiceFuture<Result<PreparedBuiltinOperands, RuntimeError>> {
        let name = request.admission().binding.builtin.name;
        assert!(NAMES.contains(&name) || name == "ind2rgb");
        assert_eq!(
            request.form().operation,
            BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather)
        );
        assert_eq!(
            request.form().data_arguments,
            if name == "ind2rgb" {
                &[0, 1][..]
            } else {
                &[0][..]
            }
        );
        self.events.borrow_mut().push("prepare");
        let mut operands = request.into_operands();
        if let Some(replacement) = &self.replacement {
            operands[0] = replacement.clone();
        }
        let prepared =
            PreparedBuiltinOperands::new(operands, Some(Box::new(Observer(self.events.clone()))));
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
    enter_with_replacement(events, pause, None)
}

fn enter_with_replacement(
    events: &Events,
    pause: bool,
    replacement: Option<Value>,
) -> RuntimeContextGuard {
    RuntimeContextGuard::enter(
        RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_acceleration(Rc::new(
                Service {
                    events: events.clone(),
                    pause,
                    replacement,
                },
            ))),
    )
}

#[test]
fn prepared_host_operand_is_consumed_once_but_result_restores_to_original_resident_owner() {
    use crate::builtins::common::{gpu_helpers, test_support};
    use runmat_value::IntegerStorage;
    test_support::with_test_provider(|provider| {
        let source_rgb =
            Tensor::new_integer(IntegerStorage::U8(vec![255, 0, 0]), vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &source_rgb)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        // The fixture substitutes a different already-materialized host operand;
        // the output witnesses that the body does not gather the original again.
        let prepared_green = Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![0, 255, 0]), vec![1, 1, 3]).unwrap(),
        );
        for name in ["rgb2gray", "rgb2hsv"] {
            let events = Events::default();
            let _context = enter_with_replacement(&events, false, Some(prepared_green.clone()));
            let result = block_on(dispatcher::call_builtin_async(
                name,
                &[Value::GpuTensor(source.clone())],
            ))
            .unwrap();
            let Value::GpuTensor(output) = result else {
                panic!("explicit result must be resident");
            };
            let owner = runmat_accelerate_api::provider_for_handle(&output).unwrap();
            assert!(runmat_accelerate_api::same_provider_instance(
                owner, provider
            ));
            let observed = test_support::gather(Value::GpuTensor(output))
                .unwrap()
                .materialize_f64();
            if name == "rgb2gray" {
                assert_eq!(observed, vec![150.0]);
            } else {
                assert!((observed[0] - 1.0 / 3.0).abs() < 1e-12);
                assert_eq!(&observed[1..], &[1.0, 1.0]);
            }
            assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
        }
        assert_eq!(
            test_support::gather(Value::GpuTensor(source))
                .unwrap()
                .integer_storage(),
            Some(&IntegerStorage::U8(vec![255, 0, 0]))
        );
    });
}

#[test]
fn prepared_hsv_operand_is_used_once_and_restores_original_owner() {
    use crate::builtins::common::{gpu_helpers, test_support};
    test_support::with_test_provider(|provider| {
        let original = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let source = gpu_helpers::upload_tensor(provider, &original)
            .unwrap()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .unwrap();
        let prepared =
            Value::Tensor(Tensor::new(vec![1.0 / 3.0, 1.0, 1.0], vec![1, 1, 3]).unwrap());
        let events = Events::default();
        let _context = enter_with_replacement(&events, false, Some(prepared));
        let result = block_on(dispatcher::call_builtin_async(
            "hsv2rgb",
            &[Value::GpuTensor(source.clone())],
        ))
        .unwrap();
        let Value::GpuTensor(output) = result else {
            panic!("explicit result must be resident");
        };
        let owner = runmat_accelerate_api::provider_for_handle(&output).unwrap();
        assert!(runmat_accelerate_api::same_provider_instance(
            owner, provider
        ));
        assert_eq!(
            test_support::gather(Value::GpuTensor(output))
                .unwrap()
                .materialize_f64(),
            vec![0.0, 1.0, 0.0]
        );
        assert_eq!(
            test_support::gather(Value::GpuTensor(source))
                .unwrap()
                .materialize_f64(),
            vec![0.0, 1.0, 1.0]
        );
        assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
    });
}

#[test]
fn indexed_color_two_operand_form_observes_once_and_invalid_host_index_does_not_prepare() {
    let events = Events::default();
    let _context = enter(&events, false);
    let map = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap());
    let valid = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
    let entry = runmat_builtins::builtin_catalog_primary_entry_by_name("ind2rgb").unwrap();
    let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
        panic!("missing form");
    };
    assert_eq!(forms[0].data_arguments, &[0, 1]);
    assert!(runmat_builtins::validate_builtin_preparation(entry).is_empty());
    let output = block_on(dispatcher::call_builtin_async(
        "ind2rgb",
        &[valid.clone(), map.clone()],
    ))
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("RGB output tensor");
    };
    assert_eq!(output.shape, vec![1, 1, 3]);
    assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
    events.borrow_mut().clear();
    let invalid = Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap());
    let error = block_on(dispatcher::call_builtin_async(
        "ind2rgb",
        &[invalid, map.clone()],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ind2rgb:InvalidIndex"));
    assert!(events.borrow().is_empty());
    let invalid_map = Value::Tensor(
        Tensor::new_with_dtype(vec![1.0, 0.0, 0.0], vec![1, 3], NumericDType::F32).unwrap(),
    );
    let error = block_on(dispatcher::call_builtin_async(
        "ind2rgb",
        &[valid, invalid_map],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ind2rgb:InvalidColormap"));
    assert!(events.borrow().is_empty());
}

#[test]
fn indexed_color_unpolled_and_pending_calls_do_not_publish_preparation() {
    let events = Events::default();
    let _context = enter(&events, true);
    let args = [
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap()),
    ];
    drop(dispatcher::call_builtin_async("ind2rgb", &args));
    assert!(events.borrow().is_empty());
    let mut call = Box::pin(dispatcher::call_builtin_async("ind2rgb", &args));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(&*events.borrow(), &["prepare"]);
    drop(call);
    assert_eq!(&*events.borrow(), &["prepare", "abort"]);
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
    let events = Events::default();
    let _context = enter(&events, false);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
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
        block_on(dispatcher::call_builtin_async_with_outputs(
            name,
            &[valid(name)],
            0,
        ))
        .unwrap();
        assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
        events.borrow_mut().clear();
    }
}

#[test]
fn extension_gate_and_empty_rgb_inputs_precede_service() {
    let events = Events::default();
    let _context = enter(&events, false);
    {
        let _disabled = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(dispatcher::call_builtin_async(
            "gray2rgb",
            &[valid("gray2rgb")],
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:Gray2rgbCallExtension")
        );
        assert!(events.borrow().is_empty());
    }
    for name in ["hsv2rgb", "rgb2gray", "rgb2hsv"] {
        let empty = Value::Tensor(Tensor::new(vec![], vec![0, 1, 3]).unwrap());
        let error = block_on(dispatcher::call_builtin_async(name, &[empty])).unwrap_err();
        let expected = format!("RunMat:{name}:InvalidInput");
        assert_eq!(error.identifier(), Some(expected.as_str()));
        assert!(events.borrow().is_empty());
    }
    let _enabled = crate::compatibility::push_runmat_extensions_enabled(true);
    let empty_gray = Value::Tensor(Tensor::new(vec![], vec![0, 2]).unwrap());
    let output = block_on(dispatcher::call_builtin_async("gray2rgb", &[empty_gray])).unwrap();
    let Value::Tensor(output) = output else {
        panic!("empty RGB remains a tensor");
    };
    assert_eq!(output.shape, vec![0, 2, 3]);
    assert!(output.is_empty());
    assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
}

#[test]
fn invalid_logical_hsv_resident_geometry_precedes_service() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let handle = provider
            .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1, 1],
                shape: &[1, 3],
            })
            .unwrap();
        let events = Events::default();
        let _context = enter(&events, false);
        provider.reset_telemetry();
        let error = block_on(dispatcher::call_builtin_async(
            "hsv2rgb",
            &[Value::GpuTensor(handle.clone())],
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:hsv2rgb:InvalidInput"));
        assert!(error.message().contains("nonempty MxNx3"));
        assert!(events.borrow().is_empty());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(handle.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn known_invalid_geometry_class_and_output_count_do_not_prepare() {
    let events = Events::default();
    let _context = enter(&events, false);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
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
        // Nx3 numeric colormaps reject uint8 for these conversion forms.
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
    let events = Events::default();
    let _context = enter(&events, false);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
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
    let events = Events::default();
    let _context = enter(&events, true);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
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
