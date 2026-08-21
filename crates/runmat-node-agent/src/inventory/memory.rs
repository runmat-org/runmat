use sysinfo::Disks;

pub(super) fn total_bytes() -> u64 {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    system.total_memory()
}

pub(super) fn scratch_bytes() -> u64 {
    Disks::new_with_refreshed_list()
        .iter()
        .map(|disk| disk.available_space())
        .max()
        .unwrap_or(0)
}
