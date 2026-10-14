use std::rc::Rc;

use runmat_execution_artifact::PrivateArtifactRoot;
use runmat_native_ffi::{NativeInterfaceArtifactBundle, NativeInterfaceArtifactManifest};

pub(crate) struct InstalledNativeInterfaces {
    _root: PrivateArtifactRoot,
}

pub(crate) fn install_admitted(
    bundle: &NativeInterfaceArtifactBundle,
    adapter: &Rc<runmat_runtime::foreign::NativeFfiAdapter>,
) -> Result<InstalledNativeInterfaces, String> {
    let mut root = PrivateArtifactRoot::new("runmat-aot-native-")?;
    adapter
        .attach_private_byte_lease(root.lease())
        .map_err(|error| format!("bind standalone native-interface bytes: {error}"))?;
    for (index, entry) in bundle.interfaces.iter().enumerate() {
        let manifest = NativeInterfaceArtifactManifest::from_canonical_bytes(&entry.manifest)
            .map_err(|error| format!("decode standalone native interface: {error}"))?;
        manifest
            .validate_current_library(&entry.library)
            .map_err(|error| format!("validate standalone native interface: {error}"))?;
        let manifest_path =
            root.write_verified(&format!("{index}/manifest.json"), &entry.manifest)?;
        let library_path = root.write_verified(&format!("{index}/library.bin"), &entry.library)?;
        adapter
            .install_prepared_artifact(library_path.as_path(), manifest_path.as_path())
            .map_err(|error| format!("install standalone native interface: {error}"))?;
    }
    Ok(InstalledNativeInterfaces { _root: root })
}
