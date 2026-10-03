//! Development-only exporter/checker for the typed public authorStudy fixture.
//! The package declares this binary with `required-features = ["fea-example-fixtures"]`.
//! Its four relative paths form one isolated fixture root. Run examples with
//! that root as the filesystem current directory; no absolute machine path is
//! embedded in the typed mesh or evidence artifacts.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use runmat_meshing_evidence::example_fixtures::{
    author_study_example_files, TETRA_PATH, TETRA_STL,
};
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("expected --write or --check")?;
    let root = args.next().ok_or("expected fixture directory")?;
    if args.next().is_some() || !matches!(mode.as_str(), "--write" | "--check") {
        return Err("usage: export_fea_author_study_fixture (--write|--check) DIRECTORY".into());
    }
    let geometry = runmat_runtime::geometry::geometry_load(TETRA_PATH, TETRA_STL.as_bytes())?;
    let files = author_study_example_files(&geometry)?;
    let root = Path::new(&root);
    if mode == "--write" {
        fs::create_dir(root)?;
    } else if !root.is_dir() {
        return Err("fixture directory does not exist".into());
    }
    if fs::symlink_metadata(root)?.file_type().is_symlink() {
        return Err("fixture directory cannot be a symbolic link".into());
    }
    let mut records = Vec::new();
    for (name, bytes) in files.entries() {
        let path = root.join(name);
        if mode == "--write" {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        } else {
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() || fs::read(&path)? != bytes
            {
                return Err(format!("typed producer and fixture bytes differ: {name}").into());
            }
        }
        records.push(serde_json::json!({
            "relative_path": name,
            "byte_length": bytes.len(),
            "sha256": format!("{:x}", Sha256::digest(bytes)),
        }));
    }
    // Extra files would escape the reviewed four-entry fixture closure.
    let mut actual = fs::read_dir(root)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    actual.sort();
    let mut expected = files.entries().map(|(name, _)| name.to_string()).to_vec();
    expected.sort();
    if actual != expected {
        return Err("fixture directory has extra or missing entries".into());
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
