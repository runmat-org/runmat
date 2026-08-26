#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;

use runmat_java::{
    discover_jvm, JavaDiscoveryRequest, JavaInvocationError, JavaSession, JavaValue, JvmConfig,
    JvmProcess,
};

fn live_session() -> Option<JavaSession> {
    let home = std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(PathBuf::from)?;
    let installation = discover_jvm(&JavaDiscoveryRequest {
        explicit_home: Some(home),
        search_system: false,
        ..JavaDiscoveryRequest::default()
    })
    .unwrap();
    let process = JvmProcess::launch(installation, &JvmConfig::default()).unwrap();
    Some(JavaSession::new(process))
}

#[test]
fn captures_java_exception_type_message_and_frames() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let error = session
        .call_static_resolved(
            "java.lang.Integer",
            "parseInt",
            &[JavaValue::String("fixture-not-an-integer".into())],
        )
        .unwrap_err();
    let JavaInvocationError::Exception(exception) = error else {
        panic!("expected a structured Java exception, got {error}");
    };
    assert_eq!(exception.class_name, "java.lang.NumberFormatException");
    assert!(exception
        .message
        .as_deref()
        .is_some_and(|message| message.contains("fixture-not-an-integer")));
    assert!(!exception.frames.is_empty());
}

#[test]
fn invokes_standard_library_and_preserves_object_identity() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };

    assert_eq!(
        session
            .call_static_resolved("java.lang.Math", "abs", &[JavaValue::Int(-17)])
            .unwrap(),
        JavaValue::Int(17)
    );

    let builder = session
        .construct_resolved("java.lang.StringBuilder", &[])
        .unwrap();
    let JavaValue::Object { handle, class_name } = builder else {
        panic!("constructor did not return a Java object");
    };
    assert_eq!(class_name, "java.lang.StringBuilder");
    let returned = session
        .call_method_resolved(handle, "append", &[JavaValue::String("fixture".into())])
        .unwrap();
    assert_eq!(
        returned,
        JavaValue::Object {
            handle,
            class_name: "java.lang.StringBuilder".into(),
        }
    );
    assert_eq!(
        session
            .call_method_resolved(handle, "toString", &[])
            .unwrap(),
        JavaValue::String("fixture".into())
    );

    assert_eq!(
        session
            .get_static_field_resolved("java.lang.Integer", "MAX_VALUE")
            .unwrap(),
        JavaValue::Int(i32::MAX)
    );
    let point = session
        .construct_resolved("java.awt.Point", &[JavaValue::Int(3), JavaValue::Int(4)])
        .unwrap();
    let JavaValue::Object { handle: point, .. } = point else {
        panic!("point constructor did not return an object");
    };
    assert_eq!(
        session.get_field_resolved(point, "x").unwrap(),
        JavaValue::Int(3)
    );
    session
        .set_field_resolved(point, "x", &JavaValue::Int(9))
        .unwrap();
    assert_eq!(
        session.get_field_resolved(point, "x").unwrap(),
        JavaValue::Int(9)
    );

    assert_eq!(
        session
            .call_static_resolved("java.lang.Integer", "valueOf", &[JavaValue::Int(31)])
            .unwrap(),
        JavaValue::Int(31)
    );
    let copied = session
        .call_static_resolved(
            "java.util.Arrays",
            "copyOf",
            &[
                JavaValue::Array {
                    component: runmat_java::JavaParameterType::Int,
                    elements: vec![JavaValue::Int(1), JavaValue::Int(2)],
                },
                JavaValue::Int(3),
            ],
        )
        .unwrap();
    assert_eq!(
        copied,
        JavaValue::Array {
            component: runmat_java::JavaParameterType::Int,
            elements: vec![JavaValue::Int(1), JavaValue::Int(2), JavaValue::Int(0)],
        }
    );

    let list = session
        .construct_resolved("java.util.ArrayList", &[])
        .unwrap();
    let JavaValue::Object { handle: list, .. } = list else {
        panic!("list constructor did not return an object");
    };
    assert_eq!(
        session
            .call_method_resolved(list, "add", &[JavaValue::String("fixture-a".into())])
            .unwrap(),
        JavaValue::Boolean(true)
    );
    session
        .call_method_resolved(list, "add", &[JavaValue::String("fixture-b".into())])
        .unwrap();
    assert_eq!(
        session.collection_elements(list).unwrap(),
        vec![
            JavaValue::String("fixture-a".into()),
            JavaValue::String("fixture-b".into()),
        ]
    );
}
