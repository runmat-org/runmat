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

const NAMES: [&str; 6] = [
    "gray2rgb", "hsv2rgb", "lab2rgb", "rgb2gray", "rgb2hsv", "rgb2lab",
];
type Events = Rc<RefCell<Vec<&'static str>>>;

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
    pause: bool,
    replacement: Option<Value>,
    map_replacement: Option<Value>,
}
impl RuntimeAccelerationService for Service {
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
        let prepared = request
            .into_operands()
            .with_unbounded_values(|mut operands| {
                if let Some(replacement) = &self.replacement {
                    operands[0] = replacement.clone();
                }
                if let Some(replacement) = &self.map_replacement {
                    operands[1] = replacement.clone();
                }
                PreparedBuiltinOperands::new(
                    operands,
                    Some(Box::new(Observer(self.events.clone()))),
                )
            })
            .and_then(std::convert::identity);
        let pause = self.pause;
        Box::pin(async move {
            if pause {
                std::future::pending::<()>().await;
            }
            prepared
        })
    }
}

fn enter(events: &Events, pause: bool) -> (RuntimeContext, RuntimeContextGuard) {
    enter_with_replacement(events, pause, None)
}

fn enter_with_replacement(
    events: &Events,
    pause: bool,
    replacement: Option<Value>,
) -> (RuntimeContext, RuntimeContextGuard) {
    enter_with_replacements(events, pause, replacement, None)
}

fn enter_with_replacements(
    events: &Events,
    pause: bool,
    replacement: Option<Value>,
    map_replacement: Option<Value>,
) -> (RuntimeContext, RuntimeContextGuard) {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(
            RuntimeServicePorts::default().with_acceleration(Rc::new(Service {
                events: events.clone(),
                pause,
                replacement,
                map_replacement,
            })),
        );
    let guard = RuntimeContextGuard::enter(runtime.clone());
    (runtime, guard)
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
            let (runtime, _context) =
                enter_with_replacement(&events, false, Some(prepared_green.clone()));
            let result = block_on(dispatcher::call_builtin_async_in_context(
                &runtime,
                name,
                &[Value::GpuTensor(source.clone())],
            ))
            .unwrap();
            let Value::GpuTensor(output) = result.value() else {
                panic!("explicit result must be resident");
            };
            let owner = runmat_accelerate_api::provider_for_handle(output).unwrap();
            assert!(runmat_accelerate_api::same_provider_instance(
                owner, provider
            ));
            let observed = test_support::gather_rooted(result)
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
        let (runtime, _context) = enter_with_replacement(&events, false, Some(prepared));
        let result = block_on(dispatcher::call_builtin_async_in_context(
            &runtime,
            "hsv2rgb",
            &[Value::GpuTensor(source.clone())],
        ))
        .unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("explicit result must be resident");
        };
        let owner = runmat_accelerate_api::provider_for_handle(output).unwrap();
        assert!(runmat_accelerate_api::same_provider_instance(
            owner, provider
        ));
        assert_eq!(
            test_support::gather_rooted(result)
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
    let (runtime, _context) = enter(&events, false);
    let map = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap());
    let valid = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
    let entry = runmat_builtins::builtin_catalog_primary_entry_by_name("ind2rgb").unwrap();
    let BuiltinOperandPreparation::Forms(forms) = entry.placement.preparation else {
        panic!("missing form");
    };
    assert_eq!(forms[0].data_arguments, &[0, 1]);
    assert!(runmat_builtins::validate_builtin_preparation(entry).is_empty());
    let output = block_on(dispatcher::call_builtin_async_in_context(
        &runtime,
        "ind2rgb",
        &[valid.clone(), map.clone()],
    ))
    .unwrap();
    let Value::Tensor(output) = output.value() else {
        panic!("RGB output tensor");
    };
    assert_eq!(output.shape(), &[1, 1, 3]);
    assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
    events.borrow_mut().clear();
    let invalid = Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).unwrap());
    let error = block_on(dispatcher::call_builtin_async_in_context(
        &runtime,
        "ind2rgb",
        &[invalid, map.clone()],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ind2rgb:InvalidIndex"));
    assert!(events.borrow().is_empty());
    let invalid_map = Value::Tensor(
        Tensor::new_with_dtype(vec![1.0, 0.0, 0.0], vec![1, 3], NumericDType::F32).unwrap(),
    );
    let error = block_on(dispatcher::call_builtin_async_in_context(
        &runtime,
        "ind2rgb",
        &[valid, invalid_map],
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:ind2rgb:InvalidColormap"));
    assert!(events.borrow().is_empty());
}

#[test]
fn scalar_index_uses_two_operand_preparation_and_prepared_resident_map_once() {
    use crate::builtins::common::{gpu_helpers, test_support};
    use runmat_value::IntValue;

    test_support::with_test_provider(|provider| {
        let original = Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let map = gpu_helpers::upload_tensor(provider, &original).unwrap();
        let prepared = Value::Tensor(Tensor::new(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap());
        for index in [
            Value::Num(1.0),
            Value::Int(IntValue::U8(0)),
            Value::Int(IntValue::U16(0)),
        ] {
            let events = Events::default();
            let (runtime, _context) =
                enter_with_replacements(&events, false, None, Some(prepared.clone()));
            provider.reset_telemetry();
            let result = block_on(dispatcher::call_builtin_async_in_context(
                &runtime,
                "ind2rgb",
                &[index, Value::GpuTensor(map.clone())],
            ))
            .unwrap();
            let Value::GpuTensor(output) = result.value() else {
                panic!("resident map must restore a resident RGB result");
            };
            assert_eq!(output.shape(), &[1, 1, 3]);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
            assert_eq!(
                test_support::gather_rooted(result)
                    .unwrap()
                    .materialize_f64(),
                vec![0.0, 1.0, 0.0]
            );
            assert!(map.allocation_lease().validate_live().is_ok());
        }

        for invalid in [Value::Num(f64::NAN), Value::Num(1.5)] {
            let events = Events::default();
            let (runtime, _context) = enter(&events, false);
            provider.reset_telemetry();
            let error = block_on(dispatcher::call_builtin_async_in_context(
                &runtime,
                "ind2rgb",
                &[invalid, Value::GpuTensor(map.clone())],
            ))
            .unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:ind2rgb:InvalidIndex"));
            assert!(events.borrow().is_empty());
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);
            assert!(map.allocation_lease().validate_live().is_ok());
        }
    });
}

#[test]
fn indexed_color_unpolled_and_pending_calls_do_not_publish_preparation() {
    let events = Events::default();
    let (runtime, _context) = enter(&events, true);
    let args = [
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap()),
    ];
    drop(dispatcher::call_builtin_async_in_context(
        &runtime, "ind2rgb", &args,
    ));
    assert!(events.borrow().is_empty());
    let mut call = Box::pin(dispatcher::call_builtin_async_in_context(
        &runtime, "ind2rgb", &args,
    ));
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
    assert_eq!(&*events.borrow(), &["prepare"]);
    drop(call);
    assert_eq!(&*events.borrow(), &["prepare", "abort"]);
}

#[test]
fn lab_output_type_is_parsed_before_preparation_and_controls_native_class() {
    let events = Events::default();
    let (runtime, _context) = enter(&events, false);
    let lab = Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 1, 3]).unwrap());
    let invalid = block_on(dispatcher::call_builtin_async_in_context(
        &runtime,
        "lab2rgb",
        &[
            lab.clone(),
            Value::String("OutputType".into()),
            Value::String("int32".into()),
        ],
    ))
    .unwrap_err();
    assert_eq!(invalid.identifier(), Some("RunMat:lab2rgb:InvalidOption"));
    assert!(events.borrow().is_empty());
    let result = block_on(dispatcher::call_builtin_async_in_context(
        &runtime,
        "lab2rgb",
        &[
            lab,
            Value::String("OutputType".into()),
            Value::String("uint8".into()),
        ],
    ))
    .unwrap();
    let Value::Tensor(result) = result.value() else {
        panic!("uint8 RGB tensor");
    };
    assert_eq!(result.numeric_dtype(), NumericDType::U8);
    assert_eq!(result.shape(), &[1, 1, 3]);
    assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
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
    let (runtime, _context) = enter(&events, false);
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
        block_on(dispatcher::call_builtin_async_in_context(
            &runtime,
            name,
            &[valid(name)],
        ))
        .unwrap();
        assert_eq!(&*events.borrow(), &["prepare", "begin", "success"]);
        events.borrow_mut().clear();
        block_on(dispatcher::call_builtin_async_with_outputs_in_context(
            &runtime,
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
    let (runtime, _context) = enter(&events, false);
    {
        let _disabled = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(dispatcher::call_builtin_async_in_context(
            &runtime,
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
    for name in ["hsv2rgb", "rgb2gray", "rgb2hsv", "rgb2lab"] {
        let empty = Value::Tensor(Tensor::new(vec![], vec![0, 1, 3]).unwrap());
        let error = block_on(dispatcher::call_builtin_async_in_context(
            &runtime,
            name,
            &[empty],
        ))
        .unwrap_err();
        let expected = format!("RunMat:{name}:InvalidInput");
        assert_eq!(error.identifier(), Some(expected.as_str()));
        assert!(events.borrow().is_empty());
    }
    let _enabled = crate::compatibility::push_runmat_extensions_enabled(true);
    let empty_gray = Value::Tensor(Tensor::new(vec![], vec![0, 2]).unwrap());
    let output = block_on(dispatcher::call_builtin_async_in_context(
        &runtime,
        "gray2rgb",
        &[empty_gray],
    ))
    .unwrap();
    let Value::Tensor(output) = output.value() else {
        panic!("empty RGB remains a tensor");
    };
    assert_eq!(output.shape(), &[0, 2, 3]);
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
        let (runtime, _context) = enter(&events, false);
        provider.reset_telemetry();
        let error = block_on(dispatcher::call_builtin_async_in_context(
            &runtime,
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
    let (runtime, _context) = enter(&events, false);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in NAMES {
        assert!(
            block_on(dispatcher::call_builtin_async_with_outputs_in_context(
                &runtime,
                name,
                &[valid(name)],
                2
            ))
            .is_err()
        );
        assert!(events.borrow().is_empty());
        let invalid = if name == "gray2rgb" {
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 1, 3]).unwrap())
        } else {
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap())
        };
        // Nx3 numeric colormaps reject uint8 except RGB-to-Lab, which accepts it.
        let invalid = if name == "gray2rgb" {
            invalid
        } else if name == "rgb2lab" {
            Value::Tensor(
                Tensor::new_with_dtype(vec![1.0; 3], vec![1, 3], NumericDType::I16).unwrap(),
            )
        } else {
            Value::Tensor(
                Tensor::new_with_dtype(vec![1.0; 3], vec![1, 3], NumericDType::U8).unwrap(),
            )
        };
        assert!(block_on(dispatcher::call_builtin_async_in_context(
            &runtime,
            name,
            &[invalid]
        ))
        .is_err());
        assert!(events.borrow().is_empty());
    }
    let invalid_option = block_on(dispatcher::call_builtin_async_in_context(
        &runtime,
        "rgb2lab",
        &[
            valid("rgb2lab"),
            Value::from("ColorSpace"),
            Value::from("unsupported"),
        ],
    ))
    .unwrap_err();
    assert_eq!(
        invalid_option.identifier(),
        Some("RunMat:rgb2lab:InvalidOption")
    );
    assert!(events.borrow().is_empty());
}

#[test]
fn unsupported_resident_metadata_rejects_before_service_or_download() {
    let events = Events::default();
    let (runtime, _context) = enter(&events, false);
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
        let error = block_on(dispatcher::call_builtin_async_in_context(
            &runtime,
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
    let (runtime, _context) = enter(&events, true);
    let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in NAMES {
        let args = [valid(name)];
        drop(dispatcher::call_builtin_async_in_context(
            &runtime, name, &args,
        ));
        assert!(events.borrow().is_empty());
        let mut call = Box::pin(dispatcher::call_builtin_async_in_context(
            &runtime, name, &args,
        ));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(&*events.borrow(), &["prepare"]);
        drop(call);
        assert_eq!(&*events.borrow(), &["prepare", "abort"]);
        events.borrow_mut().clear();
    }
}
