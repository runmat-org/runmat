//! Conservative executable dependency fingerprints. Tetrahedron source changes
//! invalidate volume/downstream work while preserving curve/surface stage receipts.
use sha2::{Digest as _, Sha256};
#[path = "meshing_dependency_sources.rs"]
mod dependencies;
#[path = "meshing_source_inventory.rs"]
pub(crate) mod source_inventory;
pub use dependencies::metadata;
use std::{
    fs,
    io::{self, Read},
    path::Path,
};

pub fn fingerprint(
    workspace: &Path,
    volume: bool,
    context: &[u8],
    watch: bool,
    metadata: Option<&serde_json::Value>,
) -> io::Result<String> {
    let Some(roots) =
        metadata.and_then(|metadata| dependencies::roots(workspace, volume, metadata))
    else {
        return Ok("unavailable".into());
    };
    let Some(package_roots) =
        metadata.and_then(|value| dependencies::package_roots(workspace, value))
    else {
        return Ok("unavailable".into());
    };
    let mut files = Vec::new();
    for root in roots {
        // These are package roots, not arbitrary same-named source directories.
        if root.ends_with("runmat-cli") || (!volume && root.ends_with("runmat-meshing/tetrahedron"))
        {
            continue;
        }
        match source_inventory::collect(&root, &package_roots) {
            Ok(inventory) => {
                if watch {
                    for path in inventory.watches {
                        println!("cargo:rerun-if-changed={}", path.display());
                    }
                }
                files.extend(inventory.files);
            }
            Err(error) if error.kind() == io::ErrorKind::Unsupported => {
                if watch {
                    println!("cargo:warning=Meshing source identity unavailable: {error}");
                }
                return Ok("unavailable".into());
            }
            Err(error) => return Err(error),
        }
    }
    for path in ["Cargo.toml", "Cargo.lock", ".cargo/config.toml"] {
        if workspace.join(path).is_file() {
            files.push(workspace.join(path));
        }
    }
    files.sort();
    let mut hash = Sha256::new();
    hash.update(b"runmat-meshing-executable-dependencies/v3\0");
    hash.update((context.len() as u64).to_be_bytes());
    hash.update(context);
    for path in files {
        if watch {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let relative = path.strip_prefix(workspace).unwrap().to_string_lossy();
        hash.update((relative.len() as u64).to_be_bytes());
        hash.update(relative.as_bytes());
        let mut input = fs::File::open(&path)?;
        hash.update(input.metadata()?.len().to_be_bytes());
        let mut block = [0_u8; 64 * 1024];
        loop {
            let count = input.read(&mut block)?;
            if count == 0 {
                break;
            }
            hash.update(&block[..count]);
        }
    }
    Ok(hex::encode(hash.finalize()))
}
