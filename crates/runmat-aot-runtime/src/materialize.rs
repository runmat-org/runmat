use std::io::Write as _;
use std::path::Path;

pub(crate) fn write_private_read_only(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create embedded artifact {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("write embedded artifact {}: {error}", path.display()))?;
    let mut permissions = file
        .metadata()
        .map_err(|error| format!("inspect embedded artifact {}: {error}", path.display()))?
        .permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(path, permissions)
        .map_err(|error| format!("seal embedded artifact {}: {error}", path.display()))?;
    Ok(())
}
