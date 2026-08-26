#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;

use runmat_java::{
    discover_jvm, JavaDiscoveryRequest, JavaSession, JavaValue, JvmConfig, JvmProcess,
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
fn invokes_standard_library_and_preserves_object_identity() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };

    assert_eq!(
        session
            .call_static("java.lang.Math", "abs", "(I)I", &[JavaValue::Int(-17)],)
            .unwrap(),
        JavaValue::Int(17)
    );

    let builder = session
        .construct("java.lang.StringBuilder", "()V", &[])
        .unwrap();
    let JavaValue::Object { handle, class_name } = builder else {
        panic!("constructor did not return a Java object");
    };
    assert_eq!(class_name, "java.lang.StringBuilder");
    let returned = session
        .call_method(
            handle,
            "append",
            "(Ljava/lang/String;)Ljava/lang/StringBuilder;",
            &[JavaValue::String("fixture".into())],
        )
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
            .call_method(handle, "toString", "()Ljava/lang/String;", &[])
            .unwrap(),
        JavaValue::String("fixture".into())
    );
}
