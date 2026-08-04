use std::ffi::OsString;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use runmat_node_agent::service::{HttpNodeControlPlane, NodeAgentService, Shutdown};
use runmat_node_agent::AgentConfig;
use windows_service::define_windows_service;
use windows_service::service::{
    ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::{
    self, ServiceControlHandlerResult, ServiceStatusHandle,
};
use windows_service::service_dispatcher;

const SERVICE_NAME: &str = "RunMatNodeAgent";
static CONFIG: OnceLock<AgentConfig> = OnceLock::new();

define_windows_service!(ffi_service_main, service_main);

pub fn dispatch(config: AgentConfig) -> anyhow::Result<()> {
    CONFIG
        .set(config)
        .map_err(|_| anyhow::anyhow!("Windows service configuration was already initialized"))?;
    service_dispatcher::start(SERVICE_NAME, ffi_service_main)?;
    Ok(())
}

fn service_main(_arguments: Vec<OsString>) {
    if let Err(error) = run_service() {
        eprintln!("RunMat node agent service failed: {error:#}");
    }
}

fn run_service() -> anyhow::Result<()> {
    let config = CONFIG
        .get()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Windows service configuration is unavailable"))?;
    let shutdown = Arc::new(Shutdown::default());
    let handler_shutdown = Arc::clone(&shutdown);
    let status_handle =
        service_control_handler::register(SERVICE_NAME, move |event| match event {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                handler_shutdown.trigger();
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        })?;

    set_status(
        &status_handle,
        ServiceState::Running,
        ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
        ServiceExitCode::Win32(0),
    )?;

    let result = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let control = Arc::new(HttpNodeControlPlane::new(config.server_url.clone())?);
            let service = NodeAgentService::load(config, control)?;
            service.run(shutdown.subscribe()).await
        });

    let exit_code = if result.is_ok() {
        ServiceExitCode::Win32(0)
    } else {
        ServiceExitCode::ServiceSpecific(1)
    };
    set_status(
        &status_handle,
        ServiceState::Stopped,
        ServiceControlAccept::empty(),
        exit_code,
    )?;
    result.map_err(Into::into)
}

fn set_status(
    handle: &ServiceStatusHandle,
    state: ServiceState,
    accepted: ServiceControlAccept,
    exit_code: ServiceExitCode,
) -> windows_service::Result<()> {
    handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: state,
        controls_accepted: accepted,
        exit_code,
        checkpoint: 0,
        wait_hint: Duration::default(),
        process_id: None,
    })
}
