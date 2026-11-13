use serde::Serialize;

#[derive(Clone, Copy)]
pub(super) struct Snapshot {
    device: u32,
    upload: u64,
    download: u64,
}

pub(super) fn snapshot() -> Option<Snapshot> {
    // Read an existing WGPU owner only: observation must not initialize one.
    let provider = runmat_accelerate::registered_wgpu_provider_for_profiling()?;
    let counters = provider.telemetry_snapshot();
    Some(Snapshot {
        device: provider.device_id(),
        upload: counters.upload_bytes,
        download: counters.download_bytes,
    })
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(super) enum Delta {
    Unavailable,
    OwnerChanged,
    CounterReset,
    ProviderWideBytes {
        device: u32,
        upload_bytes: u64,
        download_bytes: u64,
    },
}

pub(super) fn delta(before: Option<Snapshot>, after: Option<Snapshot>) -> Delta {
    let (Some(before), Some(after)) = (before, after) else {
        return Delta::Unavailable;
    };
    if before.device != after.device {
        return Delta::OwnerChanged;
    }
    match (
        after.upload.checked_sub(before.upload),
        after.download.checked_sub(before.download),
    ) {
        (Some(upload_bytes), Some(download_bytes)) => Delta::ProviderWideBytes {
            device: after.device,
            upload_bytes,
            download_bytes,
        },
        _ => Delta::CounterReset,
    }
}
