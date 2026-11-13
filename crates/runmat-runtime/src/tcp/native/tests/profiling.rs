use super::*;
use crate::context::RuntimeContext;
use runmat_execution::profiling::{CommandProfilingRoot, DeferredWorkCounts};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll};

struct SignalFirstPending<F> {
    inner: Pin<Box<F>>,
    sender: Option<mpsc::Sender<()>>,
}

impl<F: Future> Future for SignalFirstPending<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let result = self.inner.as_mut().poll(context);
        if result.is_pending() {
            if let Some(sender) = self.sender.take() {
                let _ = sender.send(());
            }
        }
        result
    }
}

fn context() -> RuntimeContext {
    RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
}

fn bind() -> RuntimeTcpBind {
    RuntimeTcpBind {
        address: "127.0.0.1".into(),
        port: 0,
        io_timeout: Some(Duration::from_secs(2)),
        byte_order: RuntimeTcpByteOrder::Little,
    }
}

#[test]
fn unpolled_and_pending_accept_have_distinct_deferred_retirement() {
    let (driver, service) = service();
    let runtime = context();
    let mut request = bind();
    request.io_timeout = None;
    let server = futures::executor::block_on(
        service.bind(RuntimeTcpInvocation::from_context(Some(&runtime)), request),
    )
    .expect("bind session server");

    let unpolled_root = CommandProfilingRoot::new();
    let unpolled_context = runtime
        .clone()
        .with_command_profiling(unpolled_root.receipt());
    drop(service.accept(
        RuntimeTcpInvocation::from_context(Some(&unpolled_context)),
        server.authority.clone(),
        None,
    ));
    assert_eq!(
        unpolled_root.seal().deferred_work,
        Some(DeferredWorkCounts::default())
    );

    let pending_root = CommandProfilingRoot::new();
    let (sender, first_pending) = mpsc::channel();
    let worker = crate::execution::native_task::spawn(
        &service.driver,
        SignalFirstPending {
            inner: Box::pin(operation::accept(
                Arc::clone(&service.registry),
                RuntimeTcpIssuer::from_context(Some(&runtime)),
                server.authority.clone(),
                None,
            )),
            sender: Some(sender),
        },
        Some(pending_root.receipt()),
    );
    first_pending
        .recv_timeout(Duration::from_secs(5))
        .expect("real accept worker reached Pending");
    worker.abort();
    assert!(driver.block_on(worker).is_err(), "aborted worker retired");
    assert_eq!(
        pending_root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            missing: 1,
            ..DeferredWorkCounts::default()
        })
    );
}

#[test]
fn completed_read_worker_does_not_consume_unpublished_read_receipt() {
    let (driver, service) = service();
    let runtime = context();
    let root = CommandProfilingRoot::new();
    let selected = runtime.clone().with_command_profiling(root.receipt());
    let server = futures::executor::block_on(
        service.bind(RuntimeTcpInvocation::from_context(Some(&selected)), bind()),
    )
    .expect("bind session server");
    let client = futures::executor::block_on(service.connect(
        RuntimeTcpInvocation::from_context(Some(&selected)),
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
        RuntimeTcpInvocation::from_context(Some(&selected)),
        server.authority.clone(),
        None,
    ))
    .expect("accept");
    assert_eq!(
        futures::executor::block_on(service.write(
            RuntimeTcpInvocation::from_context(Some(&selected)),
            client.authority.clone(),
            vec![7, 8, 9],
        ))
        .expect("write"),
        3
    );

    let before = futures::executor::block_on(service.describe_client(
        RuntimeTcpInvocation::from_context(Some(&selected)),
        accepted.authority.clone(),
    ))
    .expect("describe live client");
    assert!(before.connected);
    let unread = futures::executor::block_on(service.read(
        RuntimeTcpInvocation::from_context(Some(&selected)),
        accepted.authority.clone(),
        RuntimeTcpReadMode::Count(3),
    ))
    .expect("read");
    assert_eq!(unread.bytes(), &[7, 8, 9]);
    assert_eq!(
        root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 6,
            completed: 6,
            ..DeferredWorkCounts::default()
        }),
        "the worker is retired, but the read receipt still owns its bytes"
    );
    drop(unread);
    let status = futures::executor::block_on(service.describe_client(
        RuntimeTcpInvocation::from_context(Some(&runtime)),
        accepted.authority.clone(),
    ))
    .expect("restored unread bytes");
    assert_eq!(status.bytes_available, 3);

    // Keep the outer JoinHandle unpolled after the real worker reaches Ready.
    // Its output still owns the unpublished bytes even though worker accounting
    // has settled; dropping that output must run the existing read rollback.
    let race_root = CommandProfilingRoot::new();
    let worker = crate::execution::native_task::spawn(
        &service.driver,
        operation::read(
            Arc::clone(&service.registry),
            RuntimeTcpIssuer::from_context(Some(&runtime)),
            accepted.authority.clone(),
            RuntimeTcpReadMode::Count(3),
        ),
        Some(race_root.receipt()),
    );
    driver.block_on(async {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !worker.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("read worker reached Ready before outer join was polled");
    });
    assert_eq!(
        race_root.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            completed: 1,
            ..DeferredWorkCounts::default()
        })
    );
    drop(worker);
    let restored = futures::executor::block_on(service.describe_client(
        RuntimeTcpInvocation::from_context(Some(&runtime)),
        accepted.authority,
    ))
    .expect("unconsumed worker output rolled back");
    assert_eq!(restored.bytes_available, 3);
}
