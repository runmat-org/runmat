use runmat_filesystem as vfs;

pub(super) async fn write(
    target: &super::target::Target,
    contents: &str,
) -> Result<(), super::errors::Failure> {
    if target.create_parent {
        if let Some(parent) = target.path.parent() {
            vfs::create_dir_all_async(parent)
                .await
                .map_err(|error| super::errors::Failure::write(&target.path, &error))?;
        }
    }
    vfs::write_async(&target.path, contents.as_bytes())
        .await
        .map_err(|error| super::errors::Failure::write(&target.path, &error))
}
