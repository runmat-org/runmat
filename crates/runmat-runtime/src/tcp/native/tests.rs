use super::*;
use crate::context::{RuntimeTcpBind, RuntimeTcpByteOrder, RuntimeTcpConnect};
use std::time::Duration;

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
        RuntimeTcpIssuer::Process,
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
        RuntimeTcpIssuer::Process,
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpIssuer::Process,
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
        RuntimeTcpIssuer::Process,
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
        RuntimeTcpIssuer::Process,
        fake,
        RuntimeTcpReadMode::Count(3),
    ));
    assert!(
        matches!(forged, Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority)
    );

    assert_eq!(
        futures::executor::block_on(service.write(
            RuntimeTcpIssuer::Process,
            client.authority.clone(),
            vec![1, 2, 3],
        ))
        .expect("write"),
        3
    );
    let abandoned = futures::executor::block_on(service.read(
        RuntimeTcpIssuer::Process,
        accepted.authority.clone(),
        RuntimeTcpReadMode::Count(3),
    ))
    .expect("first read");
    assert_eq!(abandoned.bytes(), &[1, 2, 3]);
    drop(abandoned);
    let status = futures::executor::block_on(
        service.describe_client(RuntimeTcpIssuer::Process, accepted.authority.clone()),
    )
    .expect("live status");
    assert!(status.connected);
    assert_eq!(status.bytes_available, 3);
    let retried = futures::executor::block_on(service.read(
        RuntimeTcpIssuer::Process,
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
        RuntimeTcpIssuer::Process,
        accepted.authority,
        RuntimeTcpReadMode::Available,
    ));
    assert!(matches!(child, Err(error) if error.kind() == TcpTransportErrorKind::InvalidAuthority));
}

#[test]
fn status_is_non_consuming_and_available_read_has_a_fixed_entry_bound() {
    let (driver, service) = service();
    let server = futures::executor::block_on(service.bind(
        RuntimeTcpIssuer::Process,
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: Some(Duration::from_secs(2)),
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpIssuer::Process,
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
        RuntimeTcpIssuer::Process,
        server.authority,
        Some(Duration::from_secs(2)),
    ))
    .expect("accept");

    let empty = futures::executor::block_on(service.read(
        RuntimeTcpIssuer::Process,
        accepted.authority.clone(),
        RuntimeTcpReadMode::Available,
    ))
    .expect("empty available read must not wait for a peer write");
    assert!(empty.publish().is_empty());

    let payload = vec![0x5a; 8192];
    assert_eq!(
        futures::executor::block_on(service.write(
            RuntimeTcpIssuer::Process,
            client.authority,
            payload.clone(),
        ))
        .expect("write"),
        payload.len()
    );
    let observed = driver.block_on(async {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let status = service
                    .describe_client(RuntimeTcpIssuer::Process, accepted.authority.clone())
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
    let unchanged = futures::executor::block_on(
        service.describe_client(RuntimeTcpIssuer::Process, accepted.authority.clone()),
    )
    .expect("repeat status");
    assert_eq!(unchanged.bytes_available, payload.len());
    let read = futures::executor::block_on(service.read(
        RuntimeTcpIssuer::Process,
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
        RuntimeTcpIssuer::Process,
        RuntimeTcpBind {
            address: "127.0.0.1".into(),
            port: 0,
            io_timeout: None,
            byte_order: RuntimeTcpByteOrder::Little,
        },
    ))
    .expect("bind");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpIssuer::Process,
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
        RuntimeTcpIssuer::Process,
        server.authority,
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
                    .describe_client(RuntimeTcpIssuer::Process, accepted.authority.clone())
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
