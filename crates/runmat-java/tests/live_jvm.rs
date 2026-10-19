#![cfg(not(target_family = "wasm"))]

use std::path::PathBuf;
use std::process::Command;

use runmat_java::{
    discover_jvm, JavaDiscoveryRequest, JavaInvocationError, JavaSession, JavaValue, JvmConfig,
    JvmProcess,
};

fn live_session() -> Option<std::rc::Rc<JavaSession>> {
    let home = std::env::var_os("RUNMAT_TEST_JAVA_HOME").map(PathBuf::from)?;
    let installation = discover_jvm(&JavaDiscoveryRequest {
        explicit_home: Some(home),
        search_system: false,
        ..JavaDiscoveryRequest::default()
    })
    .unwrap();
    let process = JvmProcess::launch(
        installation,
        &JvmConfig {
            options: vec!["-Djava.awt.headless=true".into()],
            ..JvmConfig::default()
        },
    )
    .unwrap();
    Some(std::rc::Rc::new(JavaSession::new(process)))
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
    assert_eq!(
        session
            .call_static_resolved(
                "java.lang.String",
                "format",
                &[
                    JavaValue::String("%s-%s".into()),
                    JavaValue::String("left".into()),
                    JavaValue::String("right".into()),
                ],
            )
            .unwrap(),
        JavaValue::String("left-right".into())
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

    let wrapped = session
        .call_static_resolved(
            "java.nio.CharBuffer",
            "wrap",
            &[JavaValue::String("interface-value".into())],
        )
        .unwrap();
    let JavaValue::Object {
        handle: wrapped, ..
    } = wrapped
    else {
        panic!("interface-typed argument did not return a Java object");
    };
    assert_eq!(
        session
            .call_method_resolved(wrapped, "remaining", &[])
            .unwrap(),
        JavaValue::Int(15)
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
    let view = session
        .call_static_resolved(
            "java.util.Collections",
            "unmodifiableList",
            &[JavaValue::Object {
                handle: list,
                class_name: "java.util.ArrayList".into(),
            }],
        )
        .unwrap();
    assert!(matches!(view, JavaValue::Object { .. }));
}

#[test]
fn dynamic_classpath_add_and_remove_rebuilds_the_session_loader() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let home = PathBuf::from(std::env::var_os("RUNMAT_TEST_JAVA_HOME").unwrap());
    let compiler = home.join("bin").join(if cfg!(target_os = "windows") {
        "javac.exe"
    } else {
        "javac"
    });
    if !compiler.is_file() {
        eprintln!("live JVM fixture requires javac; test was not requested");
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let package = root.path().join("fixture/dynamic");
    std::fs::create_dir_all(&package).unwrap();
    let source = package.join("DynamicValue.java");
    let interface = package.join("NumberSource.java");
    let driver = package.join("FixtureDriver.java");
    let service_access = package.join("ServiceAccess.java");
    std::fs::write(
        &source,
        "package fixture.dynamic; public final class DynamicValue implements NumberSource { public DynamicValue() {} public int value() { return 73; } public static int staticValue() { return 73; } }",
    )
    .unwrap();
    std::fs::write(
        &interface,
        "package fixture.dynamic; public interface NumberSource { int value(); }",
    )
    .unwrap();
    std::fs::write(
        &driver,
        "package fixture.dynamic; import java.sql.*; import java.util.Properties; import java.util.logging.Logger; public final class FixtureDriver implements Driver { public Connection connect(String url, Properties info) { return null; } public boolean acceptsURL(String url) { return url != null && url.startsWith(\"jdbc:fixture:\"); } public DriverPropertyInfo[] getPropertyInfo(String url, Properties info) { return new DriverPropertyInfo[0]; } public int getMajorVersion() { return 1; } public int getMinorVersion() { return 0; } public boolean jdbcCompliant() { return false; } public Logger getParentLogger() { return Logger.getGlobal(); } }",
    )
    .unwrap();
    std::fs::write(
        &service_access,
        "package fixture.dynamic; import java.util.ServiceLoader; public final class ServiceAccess { private ServiceAccess() {} public static ServiceLoader<?> load(Class<?> type, ClassLoader loader) { return ServiceLoader.load(type, loader); } }",
    )
    .unwrap();
    assert!(Command::new(compiler)
        .args(["-d", root.path().to_str().unwrap()])
        .args([&interface, &source, &driver, &service_access])
        .status()
        .unwrap()
        .success());
    let services = root.path().join("META-INF/services");
    std::fs::create_dir_all(&services).unwrap();
    std::fs::write(
        services.join("fixture.dynamic.NumberSource"),
        "fixture.dynamic.DynamicValue\n",
    )
    .unwrap();
    std::fs::write(
        services.join("java.sql.Driver"),
        "fixture.dynamic.FixtureDriver\n",
    )
    .unwrap();

    let before = session.classpath();
    let added = session.add_dynamic_classpath(root.path()).unwrap();
    assert_eq!(added.revision, before.revision + 1);
    assert_ne!(added.identity, before.identity);
    assert_eq!(
        session
            .call_static_resolved("fixture.dynamic.DynamicValue", "staticValue", &[])
            .unwrap(),
        JavaValue::Int(73)
    );

    let thread = session
        .call_static_resolved("java.lang.Thread", "currentThread", &[])
        .unwrap();
    let JavaValue::Object { handle: thread, .. } = thread else {
        panic!("currentThread must return a Java object");
    };
    let loader = session
        .call_method_resolved(thread, "getContextClassLoader", &[])
        .unwrap();
    let JavaValue::Object { handle: loader, .. } = loader else {
        panic!("context class loader must be a Java object");
    };
    let service_class = session
        .call_method_resolved(
            loader,
            "loadClass",
            &[JavaValue::String("fixture.dynamic.NumberSource".into())],
        )
        .unwrap();
    let services = session
        .call_static_resolved(
            "fixture.dynamic.ServiceAccess",
            "load",
            &[
                service_class,
                JavaValue::Object {
                    handle: loader,
                    class_name: "java.net.URLClassLoader".into(),
                },
            ],
        )
        .unwrap();
    let JavaValue::Object {
        handle: services, ..
    } = services
    else {
        panic!("ServiceLoader.load must return a Java object");
    };
    let iterator = session
        .call_method_resolved(services, "iterator", &[])
        .unwrap();
    let JavaValue::Object {
        handle: iterator, ..
    } = iterator
    else {
        panic!("ServiceLoader.iterator must return a Java object");
    };
    assert_eq!(
        session
            .call_method_resolved(iterator, "hasNext", &[])
            .unwrap(),
        JavaValue::Boolean(true)
    );
    let provider = session.call_method_resolved(iterator, "next", &[]).unwrap();
    let JavaValue::Object {
        handle: provider, ..
    } = provider
    else {
        panic!("ServiceLoader provider must be a Java object");
    };
    assert_eq!(
        session
            .call_method_resolved(provider, "value", &[])
            .unwrap(),
        JavaValue::Int(73)
    );
    let driver_class = session
        .call_method_resolved(
            loader,
            "loadClass",
            &[JavaValue::String("java.sql.Driver".into())],
        )
        .unwrap();
    let drivers = session
        .call_static_resolved(
            "fixture.dynamic.ServiceAccess",
            "load",
            &[
                driver_class,
                JavaValue::Object {
                    handle: loader,
                    class_name: "java.net.URLClassLoader".into(),
                },
            ],
        )
        .unwrap();
    let JavaValue::Object {
        handle: drivers, ..
    } = drivers
    else {
        panic!("JDBC service loader must be a Java object");
    };
    let drivers = session
        .call_method_resolved(drivers, "iterator", &[])
        .unwrap();
    let JavaValue::Object {
        handle: drivers, ..
    } = drivers
    else {
        panic!("JDBC provider iterator must be a Java object");
    };
    let driver = session.call_method_resolved(drivers, "next", &[]).unwrap();
    let JavaValue::Object { handle: driver, .. } = driver else {
        panic!("JDBC provider must be a Java object");
    };
    assert_eq!(
        session
            .call_method_resolved(
                driver,
                "acceptsURL",
                &[JavaValue::String("jdbc:fixture:memory".into())],
            )
            .unwrap(),
        JavaValue::Boolean(true)
    );
    let removed = session.remove_dynamic_classpath(root.path()).unwrap();
    assert_eq!(removed.revision, added.revision + 1);
    assert!(removed.dynamic.is_empty());
    assert!(session
        .call_static_resolved("fixture.dynamic.DynamicValue", "staticValue", &[])
        .is_err());
}

#[test]
fn functional_interface_callbacks_reenter_on_the_originating_thread() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let callback = session
        .register_callback(|invocation| {
            let [JavaValue::Int(value)] = invocation.arguments.as_slice() else {
                return Err(JavaInvocationError::Callback(
                    "expected one Java int callback argument".into(),
                ));
            };
            Ok(JavaValue::Int(value + 2))
        })
        .unwrap();
    let stream = session
        .call_static_resolved(
            "java.util.stream.IntStream",
            "iterate",
            &[JavaValue::Int(1), callback],
        )
        .unwrap();
    let JavaValue::Object { handle: stream, .. } = stream else {
        panic!("IntStream.iterate must return a Java stream object");
    };
    let stream = session
        .call_method_resolved(stream, "limit", &[JavaValue::Long(3)])
        .unwrap();
    let JavaValue::Object { handle: stream, .. } = stream else {
        panic!("IntStream.limit must preserve a Java stream object");
    };
    assert_eq!(
        session
            .call_method_resolved(stream, "toArray", &[])
            .unwrap(),
        JavaValue::Array {
            component: runmat_java::JavaParameterType::Int,
            elements: vec![JavaValue::Int(1), JavaValue::Int(3), JavaValue::Int(5)],
        }
    );
}

#[test]
fn listener_callbacks_preserve_opaque_argument_identity_and_void_returns() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let observed = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let callback_observed = std::rc::Rc::clone(&observed);
    let callback = session
        .register_callback(move |invocation| {
            assert_eq!(invocation.method_name, "propertyChange");
            assert!(!invocation.returns_value);
            let [JavaValue::Object { class_name, .. }] = invocation.arguments.as_slice() else {
                return Err(JavaInvocationError::Callback(
                    "expected one property-change event".into(),
                ));
            };
            callback_observed.borrow_mut().push(class_name.clone());
            Ok(JavaValue::Null)
        })
        .unwrap();
    let support = session
        .construct_resolved(
            "java.beans.PropertyChangeSupport",
            &[JavaValue::String("source".into())],
        )
        .unwrap();
    let JavaValue::Object {
        handle: support, ..
    } = support
    else {
        panic!("property-change support must be an object");
    };
    session
        .call_method_resolved(
            support,
            "addPropertyChangeListener",
            std::slice::from_ref(&callback),
        )
        .unwrap();
    assert_eq!(
        session
            .call_method_resolved(
                support,
                "firePropertyChange",
                &[
                    JavaValue::String("value".into()),
                    JavaValue::Int(1),
                    JavaValue::Int(2),
                ],
            )
            .unwrap(),
        JavaValue::Null
    );
    assert_eq!(
        observed.borrow().as_slice(),
        ["java.beans.PropertyChangeEvent"]
    );
    session
        .call_method_resolved(support, "removePropertyChangeListener", &[callback])
        .unwrap();
    session
        .call_method_resolved(
            support,
            "firePropertyChange",
            &[
                JavaValue::String("value".into()),
                JavaValue::Int(2),
                JavaValue::Int(3),
            ],
        )
        .unwrap();
    assert_eq!(observed.borrow().len(), 1);
}

#[test]
fn callback_from_a_java_worker_thread_fails_with_an_affinity_error() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    let callback = session
        .register_callback(|_| Ok(JavaValue::String("done".into())))
        .unwrap();
    let future = session
        .call_static_resolved(
            "java.util.concurrent.CompletableFuture",
            "supplyAsync",
            &[callback],
        )
        .unwrap();
    let JavaValue::Object { handle: future, .. } = future else {
        panic!("supplyAsync must return a future");
    };
    let error = session
        .call_method_resolved(future, "join", &[])
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("outside its originating RunMat thread or session"));
}

#[test]
fn edt_calls_execute_on_the_awt_event_thread() {
    let Some(session) = live_session() else {
        eprintln!("RUNMAT_TEST_JAVA_HOME is unset; live JVM test was not requested");
        return;
    };
    assert_eq!(
        session
            .call_static_resolved_on_edt(
                "javax.swing.SwingUtilities",
                "isEventDispatchThread",
                &[],
            )
            .unwrap(),
        JavaValue::Boolean(true)
    );
    let builder = session
        .construct_resolved_on_edt(
            "java.lang.StringBuilder",
            &[JavaValue::String("initial".into())],
        )
        .unwrap();
    let JavaValue::Object {
        handle: builder, ..
    } = builder
    else {
        panic!("StringBuilder constructor must return an object");
    };
    session
        .call_method_resolved_on_edt(builder, "append", &[JavaValue::String("-updated".into())])
        .unwrap();
    assert_eq!(
        session
            .call_method_resolved_on_edt(builder, "toString", &[])
            .unwrap(),
        JavaValue::String("initial-updated".into())
    );
}
