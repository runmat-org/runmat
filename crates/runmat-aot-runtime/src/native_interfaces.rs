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
        crate::materialize::write_private_read_only(&manifest_path, &entry.manifest)?;
        crate::materialize::write_private_read_only(&library_path, &entry.library)?;
        adapter
            .install_prepared_artifact(&library_path, &manifest_path)
            .map_err(|error| format!("install standalone native interface: {error}"))?;
    }
    Ok(InstalledNativeInterfaces { _root: root })
}
