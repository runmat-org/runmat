//! The placement decision must preserve an operational provider failure. A
//! serial fallback would execute the loop despite the failed transfer and
//! could commit a different prefix.

use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ProviderPrecision, ThreadProviderGuard,
};
use std::sync::OnceLock;

struct FailingDownloadProvider {
    owner: TestAllocationOwner,
}

impl AccelProvider for FailingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }

    fn device_info(&self) -> String {
        "parfor failed-download fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("the fixture installs its own automatic resident input")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("parfor must use typed numeric download") })
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        assert_eq!(handle.device_id(), self.device_id());
        Box::pin(async { anyhow::bail!("injected parfor numeric download failure") })
    }
}

#[test]
fn failed_automatic_gather_preserves_provider_error_without_serial_replay() {
    static PROVIDER: OnceLock<FailingDownloadProvider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| FailingDownloadProvider {
        owner: TestAllocationOwner::new(),
    });
    let allocation = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .expect("automatic resident parfor input");
    assert!(
        !runmat_accelerate_api::handle_is_explicit(&allocation.handle),
        "automatic placement fixture must not carry explicit device intent"
    );
    let mut session = RunMatSession::with_options(false, false).expect("session init");
    let _provider = ThreadProviderGuard::set(Some(provider));
    let service = std::rc::Rc::new(InProcessMultiLabExecutionService::new(2));
    session.install_execution_services(service.clone());
    let owner = session
        .runtime_context()
        .service_ports()
        .base_workspace()
        .expect("session base-workspace service")
        .owner();
    let mut delta = runmat_runtime::workspace::WorkspaceDelta::default();
    delta
        .assign("input", runmat_value::Value::GpuTensor(allocation.handle))
        .expect("prepare automatic resident source");
    owner.apply(delta).expect("install resident source");

    // A serial replay must fail with a different identifier before reaching
    // the indexed read; a failed child frame need not publish its marker.
    let outcome = execute_text_request(
        &mut session,
        "marker = 0; out = zeros(1, 2); parfor (index = 1:2, 2); error('RunMat:Test:ParforBodyEntered', 'parfor body must not execute'); out(index) = input(index); marker = marker + 1; end;",
    )
    .expect("execute request and retain its runtime diagnostic");
    let [diagnostic] = outcome.diagnostics.as_slice() else {
        panic!(
            "failed gather must produce exactly its original runtime diagnostic: {:?}",
            outcome.diagnostics
        );
    };
    assert_eq!(diagnostic.code, "RunMat:gather:DownloadFailed");
    assert_eq!(diagnostic.severity, abi::DiagnosticSeverity::Error);
    assert!(diagnostic
        .message
        .contains("injected parfor numeric download failure"));
    assert!(
        service.submitted_revisions().is_empty(),
        "no worker was submitted before the failed placement"
    );
    assert_eq!(
        session.get_variables().unwrap().get("marker"),
        Some(&runmat_value::Value::Num(0.0)),
        "serial replay must not run the loop body"
    );
}
