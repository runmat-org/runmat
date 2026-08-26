use std::io::Write as _;
use std::path::Path;
use std::rc::Rc;

use runmat_native_ffi::{NativeInterfaceArtifactBundle, NativeInterfaceArtifactManifest};

pub(crate) struct InstalledNativeInterfaces {
    _root: tempfile::TempDir,
}

pub(crate) fn install(
    bytes: &[u8],
    adapter: &Rc<runmat_runtime::foreign::NativeFfiAdapter>,
) -> Result<InstalledNativeInterfaces, String> {
    let bundle = NativeInterfaceArtifactBundle::from_canonical_bytes(bytes)
        .map_err(|error| format!("standalone native-interface bundle is invalid: {error}"))?;
    let root = tempfile::Builder::new()
        .prefix("runmat-aot-native-")
        .tempdir()
        .map_err(|error| format!("create standalone native-interface root: {error}"))?;
    for (index, entry) in bundle.interfaces.iter().enumerate() {
        let manifest = NativeInterfaceArtifactManifest::from_canonical_bytes(&entry.manifest)
            .map_err(|error| format!("decode standalone native interface: {error}"))?;
        manifest
            .validate_current_library(&entry.library)
            .map_err(|error| format!("validate standalone native interface: {error}"))?;
        let directory = root.path().join(index.to_string());
        std::fs::create_dir(&directory)
            .map_err(|error| format!("create native-interface directory: {error}"))?;
        let manifest_path = directory.join("manifest.json");
        let library_path = directory.join("library.bin");
        write_private_read_only(&manifest_path, &entry.manifest)?;
        write_private_read_only(&library_path, &entry.library)?;
        adapter
            .install_prepared_artifact(&library_path, &manifest_path)
            .map_err(|error| format!("install standalone native interface: {error}"))?;
    }
    Ok(InstalledNativeInterfaces { _root: root })
}

fn write_private_read_only(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "create native-interface artifact {}: {error}",
                path.display()
            )
        })?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| {
            format!(
                "write native-interface artifact {}: {error}",
                path.display()
            )
        })?;
    let mut permissions = file
        .metadata()
        .map_err(|error| {
            format!(
                "inspect native-interface artifact {}: {error}",
                path.display()
            )
        })?
        .permissions();
    permissions.set_readonly(true);
    std::fs::set_permissions(path, permissions)
        .map_err(|error| format!("seal native-interface artifact {}: {error}", path.display()))?;
    Ok(())
}
