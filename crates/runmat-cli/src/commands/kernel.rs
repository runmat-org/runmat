use anyhow::{bail, Result};
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
    let _ = (
        ip,
        key,
        transport,
        signature_scheme,
        shell_port,
        iopub_port,
        stdin_port,
        control_port,
        hb_port,
        connection_file,
        timeout,
    );
    bail!("Jupyter kernel runtime has been removed from this build")
}

pub async fn execute_kernel_with_connection(connection_file: PathBuf, timeout: u64) -> Result<()> {
    let _ = (connection_file, timeout);
    bail!("Jupyter kernel runtime has been removed from this build")
}
