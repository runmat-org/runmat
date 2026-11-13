use super::*;
use crate::context::{RuntimeTcpBind, RuntimeTcpByteOrder, RuntimeTcpConnect};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

mod profiling;

struct LockCheckingWaker {
    registry: std::sync::Arc<std::sync::Mutex<Registry>>,
    unlocked: AtomicBool,
}

#[derive(Debug)]
struct ForgedRelease;

impl runmat_value::ForeignResourceRelease for ForgedRelease {
    fn release(&self, _: &runmat_value::ForeignResourceKey) {}
}

impl std::task::Wake for LockCheckingWaker {
    fn wake(self: std::sync::Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.unlocked
            .store(self.registry.try_lock().is_ok(), Ordering::SeqCst);
    }
}

fn service() -> (tokio::runtime::Runtime, NativeTcpService) {
    let driver = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("isolated native TCP test driver");
    let service = NativeTcpService::new(driver.handle().clone());
    (driver, service)
}

#[test]
fn unpolled_bind_does_not_touch_network() {
    let (_driver, service) = service();
    let initial = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("reserve loopback port");
    let port = initial.local_addr().expect("port").port();
    drop(initial);
    let pending = service.bind(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port,
            io_timeout: None,
            byte_order: RuntimeTcpByteOrder::Little,
        },
    );
    drop(pending);
    let probe =
        std::net::TcpListener::bind(("127.0.0.1", port)).expect("unpolled bind left port free");
    assert_eq!(probe.local_addr().expect("still bound").port(), port);
}

#[test]
fn forged_reference_cannot_read_live_socket_and_dropped_read_restores_bytes() {
    let (_driver, service) = service();
    let server = futures::executor::block_on(service.bind(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpConnect {
            host: "127.0.0.1".into(),
            port: server.local_addr.port(),
            connect_timeout: Some(Duration::from_secs(2)),
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("connect");
    let accepted = futures::executor::block_on(service.accept(
        RuntimeTcpInvocation::from_context(None),
        server.authority.clone(),
        Some(Duration::from_secs(2)),
    ))
    .expect("accept");

    let fake = runmat_value::ForeignRef::detached(
        accepted.authority.resource_key(),
        accepted.authority.type_identity.clone(),
        accepted.authority.ownership,
        accepted.authority.affinity,
        accepted.authority.lifetime,
    );
    let forged = futures::executor::block_on(service.read(
        RuntimeTcpInvocation::from_context(None),
        fake,
        RuntimeTcpReadMode::Count(3),
    ));
    assert!(
        matches!(forged, Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority)
    );

    assert_eq!(
        futures::executor::block_on(service.write(
            RuntimeTcpInvocation::from_context(None),
            client.authority.clone(),
            vec![1, 2, 3],
        ))
        .expect("write"),
        3
    );
    let abandoned = futures::executor::block_on(service.read(
        RuntimeTcpInvocation::from_context(None),
        accepted.authority.clone(),
        RuntimeTcpReadMode::Count(3),
    ))
    .expect("first read");
    assert_eq!(abandoned.bytes(), &[1, 2, 3]);
    drop(abandoned);
    let status = futures::executor::block_on(service.describe_client(
        RuntimeTcpInvocation::from_context(None),
        accepted.authority.clone(),
    ))
    .expect("live status");
    assert!(status.connected);
    assert_eq!(status.bytes_available, 3);
    let retried = futures::executor::block_on(service.read(
        RuntimeTcpInvocation::from_context(None),
        accepted.authority.clone(),
        RuntimeTcpReadMode::Count(3),
    ))
    .expect("retry read");
    assert_eq!(retried.publish(), vec![1, 2, 3]);

    assert!(futures::executor::block_on(
        service.close(RuntimeTcpIssuer::Process, server.authority,)
    )
    .expect("close server"));
    let child = futures::executor::block_on(service.read(
        RuntimeTcpInvocation::from_context(None),
        accepted.authority,
        RuntimeTcpReadMode::Available,
    ));
    assert!(matches!(child, Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority));
}

#[test]
fn status_is_non_consuming_and_available_read_has_a_fixed_entry_bound() {
    let (driver, service) = service();
    let server = futures::executor::block_on(service.bind(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpConnect {
            host: "127.0.0.1".into(),
            port: server.local_addr.port(),
            connect_timeout: Some(Duration::from_secs(2)),
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("connect");
    let accepted = futures::executor::block_on(service.accept(
        RuntimeTcpInvocation::from_context(None),
        server.authority.clone(),
        Some(Duration::from_secs(2)),
    ))
    .expect("accept");

    let empty = futures::executor::block_on(service.read(
        RuntimeTcpInvocation::from_context(None),
        accepted.authority.clone(),
        RuntimeTcpReadMode::Available,
    ))
    .expect("empty available read must not wait for a peer write");
    assert!(empty.publish().is_empty());

    let payload = vec![0x5a; 8192];
    assert_eq!(
        futures::executor::block_on(service.write(
            RuntimeTcpInvocation::from_context(None),
            client.authority.clone(),
            payload.clone(),
        ))
        .expect("write"),
        payload.len()
    );
    let observed = driver.block_on(async {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let status = service
                    .describe_client(
                        RuntimeTcpInvocation::from_context(None),
                        accepted.authority.clone(),
                    )
                    .await
                    .expect("status");
                if status.bytes_available == payload.len() {
                    break status;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("socket bytes observable")
    });
    assert!(observed.connected);
    let unchanged = futures::executor::block_on(service.describe_client(
        RuntimeTcpInvocation::from_context(None),
        accepted.authority.clone(),
    ))
    .expect("repeat status");
    assert_eq!(unchanged.bytes_available, payload.len());
    let read = futures::executor::block_on(service.read(
        RuntimeTcpInvocation::from_context(None),
        accepted.authority,
        RuntimeTcpReadMode::Available,
    ))
    .expect("bounded snapshot read");
    assert_eq!(read.publish(), payload);
}

#[test]
fn status_observes_peer_eof_without_consuming_socket_bytes() {
    let (driver, service) = service();
    let server = futures::executor::block_on(service.bind(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: None,
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpConnect {
            host: "127.0.0.1".into(),
            port: server.local_addr.port(),
            connect_timeout: Some(Duration::from_secs(2)),
            io_timeout: None,
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("connect");
    let accepted = futures::executor::block_on(service.accept(
        RuntimeTcpInvocation::from_context(None),
        server.authority.clone(),
        None,
    ))
    .expect("accept");
    assert!(futures::executor::block_on(
        service.close(RuntimeTcpIssuer::Process, client.authority)
    )
    .expect("close peer"));
    let status = driver.block_on(async {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let status = service
                    .describe_client(
                        RuntimeTcpInvocation::from_context(None),
                        accepted.authority.clone(),
                    )
                    .await
                    .expect("status");
                if !status.connected {
                    break status;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("EOF observed")
    });
    assert_eq!(status.bytes_available, 0);
}

#[test]
fn closing_a_server_wakes_pending_accept_without_registry_lock() {
    use std::future::Future;

    let (_driver, service) = service();
    let issuer = RuntimeTcpIssuer::Process;
    let server = futures::executor::block_on(service.bind(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: None,
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let retained = registry::server(&service.registry, &issuer, &server.authority)
        .expect("same issuer owns server");
    let mut receiver = retained.closed.subscribe();
    let mut changed = Box::pin(receiver.changed());
    let probe = std::sync::Arc::new(LockCheckingWaker {
        registry: std::sync::Arc::clone(&service.registry),
        unlocked: AtomicBool::new(false),
    });
    let waker = std::task::Waker::from(std::sync::Arc::clone(&probe));
    assert!(changed
        .as_mut()
        .poll(&mut std::task::Context::from_waker(&waker))
        .is_pending());
    assert!(registry::close(&service.registry, &issuer, &server.authority).expect("close"));
    assert!(probe.unlocked.load(Ordering::SeqCst));
}

#[test]
fn authentic_repeated_close_is_zero_and_server_scope_closes_accepted_children() {
    use crate::context::RuntimeTcpCloseScope;

    let (_driver, service) = service();
    let issuer = RuntimeTcpIssuer::Process;
    let server = futures::executor::block_on(service.bind(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpConnect {
            host: "127.0.0.1".into(),
            port: server.local_addr.port(),
            connect_timeout: Some(Duration::from_secs(2)),
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("connect");
    let accepted = futures::executor::block_on(service.accept(
        RuntimeTcpInvocation::from_context(None),
        server.authority.clone(),
        Some(Duration::from_secs(2)),
    ))
    .expect("accept");
    let forged = runmat_value::ForeignRef::detached(
        accepted.authority.resource_key(),
        accepted.authority.type_identity.clone(),
        accepted.authority.ownership,
        accepted.authority.affinity,
        accepted.authority.lifetime,
    );
    let forged_managed = runmat_value::ForeignRef::managed(
        accepted.authority.resource_key(),
        accepted.authority.type_identity.clone(),
        accepted.authority.ownership,
        accepted.authority.affinity,
        accepted.authority.lifetime,
        std::sync::Arc::new(ForgedRelease),
    );
    let changed_lifetime = runmat_value::ForeignRef::detached(
        accepted.authority.resource_key(),
        accepted.authority.type_identity.clone(),
        accepted.authority.ownership,
        accepted.authority.affinity,
        runmat_types::ForeignLifetime::Session,
    );
    let foreign_runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let foreign_issuer = RuntimeTcpIssuer::from_context(Some(&foreign_runtime));

    assert_eq!(
        futures::executor::block_on(
            service.close_scope(issuer.clone(), RuntimeTcpCloseScope::Servers)
        )
        .expect("close server and child"),
        2
    );
    assert!(
        !futures::executor::block_on(service.close(issuer.clone(), server.authority.clone()))
            .expect("authentic server was already closed")
    );
    assert!(!futures::executor::block_on(
        service.close(issuer.clone(), accepted.authority.clone())
    )
    .expect("authentic child was already closed"));
    assert!(matches!(
        futures::executor::block_on(service.close(issuer.clone(), forged)),
        Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority
    ));
    assert!(matches!(
        futures::executor::block_on(service.close(issuer.clone(), forged_managed)),
        Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority
    ));
    assert!(matches!(
        futures::executor::block_on(service.close(foreign_issuer, accepted.authority.clone())),
        Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority
    ));
    assert!(matches!(
        futures::executor::block_on(service.close(issuer.clone(), changed_lifetime)),
        Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority
    ));
    assert!(
        futures::executor::block_on(service.close(issuer.clone(), client.authority.clone()))
            .expect("independent client remains open")
    );
    assert!(
        !futures::executor::block_on(service.close(issuer, client.authority))
            .expect("independent client repeated close")
    );
}

#[test]
fn dropping_last_server_value_closes_its_accepted_child_without_losing_close_identity() {
    let (_driver, service) = service();
    let issuer = RuntimeTcpIssuer::Process;
    let server = futures::executor::block_on(service.bind(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpInvocation::from_context(None),
        RuntimeTcpConnect {
            host: "127.0.0.1".into(),
            port: server.local_addr.port(),
            connect_timeout: Some(Duration::from_secs(2)),
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("connect");
    let child = futures::executor::block_on(service.accept(
        RuntimeTcpInvocation::from_context(None),
        server.authority.clone(),
        Some(Duration::from_secs(2)),
    ))
    .expect("accept");

    drop(server);
    assert_eq!(registry::closed_count(&service.registry), 1);
    assert!(
        !futures::executor::block_on(service.close(issuer.clone(), child.authority))
            .expect("accepted child closed when its server authority expired")
    );
    assert_eq!(registry::closed_count(&service.registry), 0);
    assert!(
        futures::executor::block_on(service.close(issuer, client.authority))
            .expect("independent outgoing client remains open")
    );
}

#[test]
fn ending_one_session_releases_its_retained_socket_without_closing_sibling_session() {
    use crate::context::{RuntimeContext, RuntimeServicePorts};
    use std::rc::Rc;

    let (_driver, native) = service();
    let service = Rc::new(native);
    let runtime_a = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
    let runtime_b = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
    service
        .register_session_cleanup(&runtime_a)
        .expect("register first exact session");
    service
        .register_session_cleanup(&runtime_a)
        .expect("repeat registration is idempotent");
    service
        .register_session_cleanup(&runtime_b)
        .expect("register independent session");
    let issuer_b = RuntimeTcpIssuer::from_context(Some(&runtime_b));
    let bind = || RuntimeTcpBind {
        address: "127.0.0.1".into(),
        port: 0,
        io_timeout: None,
        byte_order: RuntimeTcpByteOrder::Little,
    };
    let server_a = futures::executor::block_on(
        service.bind(RuntimeTcpInvocation::from_context(Some(&runtime_a)), bind()),
    )
    .expect("first session server");
    let server_b = futures::executor::block_on(
        service.bind(RuntimeTcpInvocation::from_context(Some(&runtime_b)), bind()),
    )
    .expect("sibling session server");
    assert_eq!(registry::active_count(&service.registry), 2);
    drop(runtime_a);
    assert_eq!(registry::active_count(&service.registry), 1);
    assert!(matches!(
        futures::executor::block_on(service.close(
            RuntimeTcpIssuer::from_context(Some(&runtime_b)),
            server_a.authority
        )),
        Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority
    ));
    assert!(registry::server(&service.registry, &issuer_b, &server_b.authority).is_ok());
    assert!(
        futures::executor::block_on(service.close(issuer_b, server_b.authority))
            .expect("sibling session still owns its server")
    );
}
