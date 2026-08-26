#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::rc::Rc;

use runmat_java::{JavaDiscoveryRequest, JvmConfig, JAVA_ADAPTER_ID};
use runmat_runtime::context::{ForeignCall, RuntimeContext, RuntimeServicePorts};
use runmat_runtime::execution::RuntimeExecutionService;
use runmat_runtime::foreign::{ForeignPlatform, ForeignRuntime, JavaAdapter};
use runmat_value::{IntValue, Value};

fn live_runtime() -> Option<(Rc<ForeignRuntime>, RuntimeContext)> {
    let home = std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(PathBuf::from)?;
    let foreign = Rc::new(ForeignRuntime::new(ForeignPlatform::Native));
    let adapter = JavaAdapter::with_configuration(
        foreign.handles().clone(),
        JavaDiscoveryRequest {
            explicit_home: Some(home),
            search_system: false,
            ..JavaDiscoveryRequest::default()
        },
        JvmConfig::default(),
    )
    .expect("register Java host");
    foreign
        .register_adapter(adapter)
        .expect("register Java adapter");
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_foreign(foreign.clone()));
    Some((foreign, context))
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
    let Some((_foreign, context)) = live_runtime() else {
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
    let Some((_foreign, context)) = live_runtime() else {
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
