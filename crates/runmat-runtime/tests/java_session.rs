#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;
use std::{cell::RefCell, future::Future};

use runmat_java::{JavaDiscoveryRequest, JvmConfig, JAVA_ADAPTER_ID};
use runmat_runtime::context::{
    ForeignCall, RuntimeCallRequest, RuntimeCallService, RuntimeContext, RuntimeServicePorts,
};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::foreign::{ForeignPlatform, ForeignRuntime, JavaAdapter};
use runmat_value::{CellArray, IntValue, IntegerStorage, Tensor, Value};

#[derive(Default)]
struct IncrementingCallService {
    requests: RefCell<Vec<RuntimeCallRequest>>,
}

impl RuntimeCallService for IncrementingCallService {
    fn resolve(&self, name: &str) -> Option<usize> {
        (name == "increment_callback").then_some(1)
    }

    fn invoke(
        &self,
        request: RuntimeCallRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value, runmat_runtime::RuntimeError>> + 'static>> {
        let result = match request.arguments.as_slice() {
            [Value::Int(IntValue::I32(value))] => Value::Int(IntValue::I32(value + 2)),
            _ => Value::Num(f64::NAN),
        };
        self.requests.borrow_mut().push(request);
        Box::pin(async move { Ok(result) })
    }
}

fn live_runtime() -> Option<(Rc<ForeignRuntime>, RuntimeContext, Rc<JavaAdapter>)> {
    let home = std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(PathBuf::from)?;
    let foreign = Rc::new(ForeignRuntime::new(ForeignPlatform::Native));
    let adapter = JavaAdapter::with_configuration(
        foreign.handles().clone(),
        JavaDiscoveryRequest {
            explicit_home: Some(home),
            search_system: false,
            ..JavaDiscoveryRequest::default()
        },
        JvmConfig {
            options: vec!["-Djava.awt.headless=true".into()],
            ..JvmConfig::default()
        },
    )
    .expect("register Java host");
    foreign
        .register_adapter(adapter.clone())
        .expect("register Java adapter");
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_foreign(foreign.clone()));
    Some((foreign, context, adapter))
}

fn invoke(
    context: &RuntimeContext,
    symbol: &str,
    arguments: Vec<Value>,
) -> Result<Value, Box<runmat_runtime::RuntimeError>> {
    let service = context
        .service_ports()
        .require_foreign(symbol)
        .expect("foreign service")
        .clone();
    futures::executor::block_on(context.scope(service.invoke(
        context.clone(),
        ForeignCall {
            adapter: JAVA_ADAPTER_ID.into(),
            symbol: symbol.into(),
            arguments,
            requested_outputs: 1,
        },
    )))
    .map_err(Box::new)
}

#[test]
fn java_objects_round_trip_through_runtime_owned_foreign_handles() {
    let Some((_foreign, context, _adapter)) = live_runtime() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };

    let object = invoke(
        &context,
        "construct",
        vec![Value::String("java.lang.StringBuilder".into())],
    )
    .expect("construct Java object");
    let Value::Foreign(reference) = object else {
        panic!("Java constructor must return a foreign reference");
    };
    let returned = invoke(
        &context,
        "invoke_member",
        vec![
            Value::Foreign(reference.clone()),
            Value::String("append".into()),
            Value::String("runtime-fixture".into()),
        ],
    )
    .expect("invoke Java member");
    let Value::Foreign(returned) = returned else {
        panic!("identity-returning Java method must return a foreign reference");
    };
    assert!(reference.is_same_resource(&returned));
    assert_eq!(
        invoke(
            &context,
            "invoke_member",
            vec![Value::Foreign(returned), Value::String("length".into()),],
        )
        .expect("invoke Java length"),
        Value::Int(IntValue::I32(15))
    );
}

#[test]
fn java_exceptions_keep_their_public_runtime_identifier() {
    let Some((_foreign, context, _adapter)) = live_runtime() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let error = invoke(
        &context,
        "call_static",
        vec![
            Value::String("java.lang.Integer".into()),
            Value::String("parseInt".into()),
            Value::String("not-a-number".into()),
        ],
    )
    .expect_err("invalid integer text must raise a Java exception");
    assert_eq!(error.identifier(), Some("RunMat:Java:Exception"));
    assert!(error
        .to_string()
        .contains("java.lang.NumberFormatException"));
}

#[test]
fn wide_unsigned_values_and_java_arrays_cross_the_runtime_boundary_exactly() {
    let Some((_foreign, context, _adapter)) = live_runtime() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    assert_eq!(
        invoke(
            &context,
            "call_static",
            vec![
                Value::String("java.lang.String".into()),
                Value::String("valueOf".into()),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
        )
        .expect("convert exact uint64 through BigInteger"),
        Value::String(u64::MAX.to_string())
    );

    let array = invoke(
        &context,
        "new_array",
        vec![
            Value::String("java.lang.String".into()),
            Value::Num(2.0),
            Value::Int(IntValue::U8(3)),
        ],
    )
    .expect("create multidimensional Java array");
    let Value::Foreign(array) = array else {
        panic!("javaArray must preserve the opaque Java array identity");
    };
    assert_eq!(
        invoke(
            &context,
            "call_static",
            vec![
                Value::String("java.lang.reflect.Array".into()),
                Value::String("getLength".into()),
                Value::Foreign(array),
            ],
        )
        .expect("read Java array length"),
        Value::Int(IntValue::I32(2))
    );
}

#[test]
fn typed_vectors_and_reference_collections_use_reviewed_java_copies() {
    let Some((_foreign, context, _adapter)) = live_runtime() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let copied = invoke(
        &context,
        "call_static",
        vec![
            Value::String("java.util.Arrays".into()),
            Value::String("copyOf".into()),
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I32(vec![3, 5]), vec![1, 2]).unwrap(),
            ),
            Value::Int(IntValue::I32(3)),
        ],
    )
    .expect("copy an exact integer vector through Java");
    let Value::Tensor(copied) = copied else {
        panic!("Java int[] must return an integer tensor");
    };
    assert_eq!(
        copied.integer_storage(),
        Some(&IntegerStorage::I32(vec![3, 5, 0]))
    );

    let rendered = invoke(
        &context,
        "call_static",
        vec![
            Value::String("java.util.Arrays".into()),
            Value::String("toString".into()),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("entry".into()), Value::Int(IntValue::I32(8))],
                    1,
                    2,
                )
                .unwrap(),
            ),
        ],
    )
    .expect("box a heterogeneous cell as Object[]");
    assert_eq!(rendered, Value::String("[entry, 8]".into()));

    let pattern = invoke(
        &context,
        "call_static",
        vec![
            Value::String("java.util.regex.Pattern".into()),
            Value::String("compile".into()),
            Value::String(",".into()),
        ],
    )
    .expect("compile a regular expression");
    let Value::Foreign(pattern) = pattern else {
        panic!("Pattern.compile must return a Java object");
    };
    let split = invoke(
        &context,
        "invoke_member",
        vec![
            Value::Foreign(pattern),
            Value::String("split".into()),
            Value::String("left,right".into()),
        ],
    )
    .expect("convert a returned Java String array");
    let Value::StringArray(split) = split else {
        panic!("Java String[] must return a RunMat string array");
    };
    assert_eq!(split.data, vec!["left", "right"]);
}

#[test]
fn java_environment_configuration_and_desktop_capability_are_session_owned() {
    let Some((_foreign, context, adapter)) = live_runtime() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    assert_eq!(
        invoke(&context, "usejava", vec![Value::String("desktop".into())],).unwrap(),
        Value::Bool(false)
    );
    adapter.set_desktop_available(true);
    assert_eq!(
        invoke(&context, "usejava", vec![Value::String("desktop".into())],).unwrap(),
        Value::Bool(true)
    );

    let status = invoke(
        &context,
        "configure",
        vec![
            Value::String("Version".into()),
            Value::String(
                std::env::var_os("RUNMAT_TEST_JAVA_HOME")
                    .expect("live Java home")
                    .to_string_lossy()
                    .into_owned(),
            ),
        ],
    )
    .expect("pin the Java major version before startup");
    let Value::Object(status) = status else {
        panic!("jenv configuration must return a JavaEnvironment object");
    };
    assert_eq!(status.class_name, "matlab.javaclient.JavaEnvironment");
    assert_eq!(
        status.properties.get("Status"),
        Some(&Value::String("notloaded".into()))
    );
}

#[test]
fn java_callbacks_reenter_the_originating_runtime_call_service() {
    let Some((foreign, base_context, _adapter)) = live_runtime() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let calls = Rc::new(IncrementingCallService::default());
    let context = base_context.with_service_ports(
        RuntimeServicePorts::default()
            .with_foreign(foreign)
            .with_call(calls.clone()),
    );
    let stream = invoke(
        &context,
        "call_static",
        vec![
            Value::String("java.util.stream.IntStream".into()),
            Value::String("iterate".into()),
            Value::Int(IntValue::I32(1)),
            Value::FunctionHandle("increment_callback".into()),
        ],
    )
    .expect("create a stream with a RunMat callback");
    let Value::Foreign(stream) = stream else {
        panic!("IntStream.iterate must return a Java stream");
    };
    let stream = invoke(
        &context,
        "invoke_member",
        vec![
            Value::Foreign(stream),
            Value::String("limit".into()),
            Value::Int(IntValue::I64(3)),
        ],
    )
    .expect("limit the stream");
    let Value::Foreign(stream) = stream else {
        panic!("IntStream.limit must preserve the stream");
    };
    let values = invoke(
        &context,
        "invoke_member",
        vec![Value::Foreign(stream), Value::String("toArray".into())],
    )
    .expect("evaluate the callback-backed stream");
    let Value::Tensor(values) = values else {
        panic!("IntStream.toArray must return an integer tensor");
    };
    assert_eq!(
        values.integer_storage(),
        Some(&IntegerStorage::I32(vec![1, 3, 5]))
    );
    let requests = calls.requests.borrow();
    assert_eq!(requests.len(), 2);
    assert!(requests
        .iter()
        .all(|request| request.requested_outputs == 1));
    assert_eq!(requests[0].arguments, vec![Value::Int(IntValue::I32(1))]);
}

#[test]
fn java_edt_operations_require_desktop_authority_and_run_on_the_edt() {
    let Some((_foreign, context, adapter)) = live_runtime() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let error = invoke(
        &context,
        "call_static_edt",
        vec![
            Value::String("javax.swing.SwingUtilities".into()),
            Value::String("isEventDispatchThread".into()),
        ],
    )
    .expect_err("headless sessions must not claim Java EDT authority");
    assert_eq!(error.identifier(), Some("RunMat:Java:EdtUnavailable"));

    adapter.set_desktop_available(true);
    assert_eq!(
        invoke(
            &context,
            "call_static_edt",
            vec![
                Value::String("javax.swing.SwingUtilities".into()),
                Value::String("isEventDispatchThread".into()),
            ],
        )
        .expect("Desktop-authorized call must run on the Java EDT"),
        Value::Bool(true)
    );
}
