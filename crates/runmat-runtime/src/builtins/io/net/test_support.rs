//! Network-domain provider witness for pre-effect admission tests.

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostTensorView, ProviderPrecision,
};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::context::{
    RuntimeContext, RuntimeServicePorts, RuntimeTcpByteOrder, RuntimeTcpConnect,
    RuntimeTcpInvocation, RuntimeTcpIssuer, RuntimeTcpService,
};
use crate::tcp::NativeTcpService;
use runmat_value::{ObjectInstance, Value};
use std::time::Duration;

#[path = "test_support/cpu_turn.rs"]
pub(crate) mod cpu_turn;

static TCP_TEST_GUARD: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    TCP_TEST_GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Test-owned host and session: no process-global socket ID or implicit port.
pub(crate) struct TcpTestHost {
    _driver: tokio::runtime::Runtime,
    service: Rc<NativeTcpService>,
    pub(crate) runtime: RuntimeContext,
}

impl TcpTestHost {
    pub(crate) fn new() -> Self {
        let driver = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("isolated TCP host driver");
        let service = Rc::new(NativeTcpService::new(driver.handle().clone()));
        let runtime =
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(RuntimeServicePorts::default().with_tcp(service.clone()));
        Self {
            _driver: driver,
            service,
            runtime,
        }
    }

    pub(crate) fn connect(&self, port: u16, timeout: f64, order: RuntimeTcpByteOrder) -> Value {
        let endpoint = futures::executor::block_on(
            self.service.connect(
                RuntimeTcpInvocation::from_context(Some(&self.runtime)),
                RuntimeTcpConnect {
                    host: "127.0.0.1".into(),
                    port,
                    connect_timeout: Some(Duration::from_secs(2)),
                    io_timeout: (timeout.is_finite() && timeout > 0.0)
                        .then(|| Duration::from_secs_f64(timeout)),
                    byte_order: order,
                },
            ),
        )
        .expect("connect test client");
        Value::Object(
            ObjectInstance::new(runmat_builtins::TCPCLIENT_CLASS)
                .with_tcp_authority(endpoint.authority),
        )
    }

    pub(crate) fn close(&self, client: &Value) {
        let authority = super::receiver::authority(client, runmat_builtins::TCPCLIENT_CLASS)
            .expect("test client authority");
        assert!(futures::executor::block_on(self.service.close(
            RuntimeTcpIssuer::from_context(Some(&self.runtime)),
            authority,
        ))
        .expect("close test client"));
    }

    pub(crate) fn wait_available(&self, client: &Value, at_least: usize) {
        let authority = super::receiver::authority(client, runmat_builtins::TCPCLIENT_CLASS)
            .expect("test client authority");
        self._driver.block_on(async {
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let status = self
                        .service
                        .describe_client(
                            RuntimeTcpInvocation::from_context(Some(&self.runtime)),
                            authority.clone(),
                        )
                        .await
                        .expect("client status");
                    if status.bytes_available >= at_least {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("test payload arrival");
        });
    }

    pub(crate) fn wait_disconnected(&self, client: &Value) {
        let authority = super::receiver::authority(client, runmat_builtins::TCPCLIENT_CLASS)
            .expect("test client authority");
        self._driver.block_on(async {
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let status = self
                        .service
                        .describe_client(
                            RuntimeTcpInvocation::from_context(Some(&self.runtime)),
                            authority.clone(),
                        )
                        .await
                        .expect("client status");
                    if !status.connected {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("peer EOF observation");
        });
    }
}

pub(crate) struct CountingDownloadProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl CountingDownloadProvider {
    pub(crate) fn new() -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }
    }

    pub(crate) fn resident(&self) -> GpuTensorHandle {
        self.upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .expect("authentic network provider allocation")
    }

    pub(crate) fn downloads(&self) -> usize {
        self.downloads.load(Ordering::SeqCst)
    }
}

impl AccelProvider for CountingDownloadProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
}
