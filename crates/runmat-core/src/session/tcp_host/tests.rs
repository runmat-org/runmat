use super::*;
use runmat_runtime::context::{RuntimeContext, RuntimeTcpByteOrder, RuntimeTcpInvocation};

fn runtime() -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
}

#[test]
fn cleanup_registration_initializes_once_without_retaining_session() {
    let service = LazyNativeTcpService::default();
    assert!(service.service.get().is_none());
    let runtime = runtime();
    let RuntimeTcpIssuer::Session { lease, .. } = RuntimeTcpIssuer::from_context(Some(&runtime))
    else {
        panic!("context-backed TCP issuer must carry a session lease");
    };
    service
        .register_session_cleanup(&runtime)
        .expect("register session before its first socket operation");
    let native = service.service.get().expect("initialized native owner");
    service
        .register_session_cleanup(&runtime)
        .expect("idempotent registration");
    assert!(std::ptr::eq(
        native,
        service.service.get().expect("same native owner")
    ));
    drop(runtime);
    assert!(
        lease.upgrade().is_none(),
        "service must not retain its session"
    );
}

#[test]
fn lazy_owner_teardown_releases_one_sessions_socket_without_closing_sibling() {
    let service = LazyNativeTcpService::default();
    let first = runtime();
    let sibling = runtime();
    service.register_session_cleanup(&first).unwrap();
    service.register_session_cleanup(&sibling).unwrap();
    let bind = || RuntimeTcpBind {
        address: "127.0.0.1".into(),
        port: 0,
        io_timeout: None,
        byte_order: RuntimeTcpByteOrder::Little,
    };
    let first_server = futures::executor::block_on(
        service.bind(RuntimeTcpInvocation::from_context(Some(&first)), bind()),
    )
    .expect("first session socket");
    let sibling_issuer = RuntimeTcpIssuer::from_context(Some(&sibling));
    let sibling_server = futures::executor::block_on(
        service.bind(RuntimeTcpInvocation::from_context(Some(&sibling)), bind()),
    )
    .expect("sibling session socket");
    drop(first);
    let _rebound = std::net::TcpListener::bind(first_server.local_addr)
        .expect("teardown releases socket despite retained authority value");
    assert!(
        futures::executor::block_on(service.close(sibling_issuer, sibling_server.authority))
            .expect("sibling socket remains owned and open")
    );
}
