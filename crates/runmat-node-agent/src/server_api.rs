//! Checked wire projections shared by bootstrap and the running node service.
use runmat_execution_transport_native::control::NodeInventory;
use runmat_execution_transport_native::{TransportError, TransportResult};
use runmat_server_client::public_api::{self, types};

pub(crate) fn inventory_to_api(
    value: NodeInventory,
) -> TransportResult<types::ResourceInventoryBody> {
    Ok(types::ResourceInventoryBody {
        cpu_millicores: to_i64(value.cpu_millicores, "CPU inventory")?,
        memory_bytes: to_i64(value.memory_bytes, "memory inventory")?,
        scratch_bytes: to_i64(value.scratch_bytes, "scratch inventory")?,
        accelerators: value
            .accelerators
            .into_iter()
            .map(runmat_server_client::execution::accelerator_device_to_api)
            .collect::<Result<_, _>>()
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
        host: runmat_server_client::execution::execution_host_inventory_to_api(value.host)
            .map_err(|error| TransportError::MalformedFrame(error.to_string()))?,
        capabilities: value.capabilities.into_iter().collect(),
    })
}

pub(crate) fn to_i64(value: u64, field: &str) -> TransportResult<i64> {
    i64::try_from(value)
        .map_err(|_| TransportError::Unavailable(format!("{field} exceeds the wire range")))
}

pub(crate) fn to_u64(value: i64, field: &str) -> TransportResult<u64> {
    u64::try_from(value).map_err(|_| TransportError::Unavailable(format!("{field} is negative")))
}

pub(crate) fn map_error<E: std::fmt::Debug>(error: public_api::Error<E>) -> TransportError {
    if error
        .status()
        .is_some_and(|status| matches!(status.as_u16(), 401 | 403))
    {
        TransportError::StaleAuthority
    } else {
        TransportError::Unavailable(error.to_string())
    }
}
