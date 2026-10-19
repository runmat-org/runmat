//! Network I/O must progress independently of synchronous work in a calling executor.
//! One bounded process-wide reactor services Quinn only; it schedules no portable tasks.
use crate::{TransportError, TransportResult};
use quinn::{AsyncTimer, AsyncUdpSocket, Runtime};
use std::{
    future::Future,
    io,
    pin::Pin,
    sync::{Arc, OnceLock},
    time::Instant,
};
use tokio::runtime::{Builder, Handle, Runtime as Executor};

const IO_THREADS: usize = 1;
const IO_THREAD_STACK_BYTES: usize = 2 * 1024 * 1024;
static EXECUTOR: OnceLock<io::Result<Executor>> = OnceLock::new();

#[derive(Debug)]
struct NetworkRuntime(Handle);

pub(super) fn shared() -> TransportResult<Arc<dyn Runtime>> {
    let executor = EXECUTOR.get_or_init(|| {
        Builder::new_multi_thread()
            .worker_threads(IO_THREADS)
            .max_blocking_threads(1)
            .thread_stack_size(IO_THREAD_STACK_BYTES)
            .thread_name("runmat-quic-io")
            .enable_all()
            .build()
    });
    let executor = executor.as_ref().map_err(|error| {
        TransportError::Unavailable(format!("QUIC network executor could not start: {error}"))
    })?;
    Ok(Arc::new(NetworkRuntime(executor.handle().clone())))
}

impl Runtime for NetworkRuntime {
    fn spawn(&self, future: Pin<Box<dyn Future<Output = ()> + Send>>) {
        self.0.spawn(future);
    }
    fn new_timer(&self, at: Instant) -> Pin<Box<dyn AsyncTimer>> {
        let _context = self.0.enter();
        quinn::TokioRuntime.new_timer(at)
    }
    fn wrap_udp_socket(&self, socket: std::net::UdpSocket) -> io::Result<Arc<dyn AsyncUdpSocket>> {
        let _context = self.0.enter();
        quinn::TokioRuntime.wrap_udp_socket(socket)
    }
}
