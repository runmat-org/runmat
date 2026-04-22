use anyhow::{Context, Result};
use runmat_kernel::{ConnectionInfo, KernelConfig, KernelServer};
use std::path::PathBuf;

#[allow(clippy::too_many_arguments)]
pub async fn execute_kernel(
    ip: String,
    key: Option<String>,
    transport: String,
    signature_scheme: String,
    shell_port: u16,
    iopub_port: u16,
    stdin_port: u16,
    control_port: u16,
    hb_port: u16,
    connection_file: Option<PathBuf>,
    timeout: u64,
) -> Result<()> {
    log::info!("Starting RunMat Jupyter kernel");

    let mut connection = ConnectionInfo {
        ip,
        transport,
        signature_scheme,
        key: key.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        shell_port,
        iopub_port,
        stdin_port,
        control_port,
        hb_port,
    };

    if shell_port == 0 || iopub_port == 0 || stdin_port == 0 || control_port == 0 || hb_port == 0 {
        connection
            .assign_ports()
            .context("Failed to assign kernel ports")?;
    }

    if let Some(path) = connection_file {
        connection
            .write_to_file(&path)
            .with_context(|| format!("Failed to write connection file to {path:?}"))?;
        log::info!("Connection file written to {path:?}");
    }

    let config = KernelConfig {
        connection,
        session_id: uuid::Uuid::new_v4().to_string(),
        debug: log::log_enabled!(log::Level::Debug),
        execution_timeout: Some(timeout),
    };

    let mut server = KernelServer::new(config);

    log::info!("Starting kernel server...");
    server
        .start()
        .await
        .context("Failed to start kernel server")?;

    log::info!("Kernel is ready. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c()
        .await
        .context("Failed to listen for ctrl-c")?;

    log::info!("Shutting down kernel...");
    server
        .stop()
        .await
        .context("Failed to stop kernel server")?;

    Ok(())
}

pub async fn execute_kernel_with_connection(connection_file: PathBuf, timeout: u64) -> Result<()> {
    log::info!("Starting kernel with connection file: {connection_file:?}");

    let connection = ConnectionInfo::from_file(&connection_file)
        .with_context(|| format!("Failed to load connection file: {connection_file:?}"))?;

    let config = KernelConfig {
        connection,
        session_id: uuid::Uuid::new_v4().to_string(),
        debug: log::log_enabled!(log::Level::Debug),
        execution_timeout: Some(timeout),
    };

    let mut server = KernelServer::new(config);

    server
        .start()
        .await
        .context("Failed to start kernel server")?;

    tokio::signal::ctrl_c()
        .await
        .context("Failed to listen for ctrl-c")?;

    server
        .stop()
        .await
        .context("Failed to stop kernel server")?;

    Ok(())
}
