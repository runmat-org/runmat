use crate::cli::PackageCacheCommand;
use anyhow::{Context, Result};
use runmat_package_cache::{CacheBackend, CacheStatus, GcPolicy};
use runmat_package_cache_native::{gc, NativeCacheConfig, SqliteCacheBackend};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) async fn execute(command: PackageCacheCommand) -> Result<()> {
    let config = NativeCacheConfig::platform_default()
        .context("failed to locate the platform package cache")?;
    let backend = SqliteCacheBackend::open(&config).context("failed to open the package cache")?;
    match command {
        PackageCacheCommand::Status { json } => {
            let snapshot = backend.snapshot().await?;
            let status = CacheStatus::from_state(&snapshot.state);
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!("Objects: {}", status.object_count);
                println!("Stored payload bytes: {}", status.stored_payload_bytes);
                println!("Logical bytes: {}", status.logical_bytes);
                println!("Pins: {}", status.pin_count);
                println!("Active leases: {}", status.lease_count);
                println!("Corruption records: {}", status.corruption_count);
            }
        }
        PackageCacheCommand::Gc { target_bytes } => {
            let snapshot = backend.snapshot().await?;
            let target = if target_bytes == 0 {
                snapshot.state.total_stored_payload_bytes() / 4
            } else {
                target_bytes
            };
            let plan = gc::execute(&backend, GcPolicy::reclaim_to(now_ms(), target), 16).await?;
            println!(
                "Reclaimed {} bytes across {} cache objects",
                plan.reclaim_bytes,
                plan.delete.len()
            );
        }
        PackageCacheCommand::Prune => {
            let plan = gc::execute(&backend, GcPolicy::reclaim_to(now_ms(), u64::MAX), 16).await?;
            println!(
                "Pruned {} bytes across {} unprotected cache objects",
                plan.reclaim_bytes,
                plan.delete.len()
            );
        }
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
