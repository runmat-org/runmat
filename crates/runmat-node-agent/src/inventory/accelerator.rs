use std::sync::OnceLock;

use runmat_execution::resource::AcceleratorDevice;

use crate::{AgentError, AgentResult};

pub(super) fn inventory() -> AgentResult<Vec<AcceleratorDevice>> {
    let mut devices = discovered_inventory()?;
    if let Some(encoded) = std::env::var("RUNMAT_NODE_ACCELERATORS")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        devices.extend(
            serde_json::from_str::<Vec<AcceleratorDevice>>(&encoded)
                .map_err(|error| AgentError::Configuration(error.to_string()))?,
        );
    }
    devices.sort_by(|left, right| left.id.cmp(&right.id));
    if devices.windows(2).any(|pair| pair[0].id >= pair[1].id) {
        return Err(AgentError::Configuration(
            "accelerator inventory contains duplicate device identities".into(),
        ));
    }
    for device in &devices {
        device
            .validate()
            .map_err(|error| AgentError::Configuration(error.to_string()))?;
    }
    Ok(devices)
}

fn discovered_inventory() -> AgentResult<Vec<AcceleratorDevice>> {
    runmat_accelerate_api::registered_providers()
        .into_iter()
        .map(|provider| provider.execution_accelerator_device(inventory_epoch()))
        .filter_map(|result| result.transpose())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| AgentError::Configuration(error.to_string()))
}

fn inventory_epoch() -> u64 {
    static EPOCH: OnceLock<u64> = OnceLock::new();
    *EPOCH.get_or_init(|| {
        let elapsed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let epoch =
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX) ^ u64::from(std::process::id());
        epoch.max(1)
    })
}
