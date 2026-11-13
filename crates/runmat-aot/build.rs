use std::{env, fs, path::PathBuf};

fn main() {
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        // This build-time packager does not ship a debugger database. Rust's
        // MSVC linker requests a PDB even with strip=debuginfo; disable it only
        // for this executable, leaving archive and consumer symbols intact.
        println!("cargo:rustc-link-arg-bin=runmat-aot-pack=/DEBUG:NONE");
    }
    const ARCHIVE_ENV: &str = "RUNMAT_AOT_RUNTIME_ARCHIVE";
    const MANIFEST_ENV: &str = "RUNMAT_AOT_RUNTIME_MANIFEST";
    println!("cargo:rerun-if-env-changed={ARCHIVE_ENV}");
    println!("cargo:rerun-if-env-changed={MANIFEST_ENV}");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    let archive = env::var_os(ARCHIVE_ENV);
    let manifest = env::var_os(MANIFEST_ENV);
    let generated = match (archive, manifest) {
        (None, None) => {
            "pub static PAYLOAD: Option<&[u8]> = None;\npub static MANIFEST: Option<&str> = None;\n"
                .to_string()
        }
        (Some(archive), Some(manifest)) => {
            let archive = PathBuf::from(archive);
            let manifest = PathBuf::from(manifest);
            println!("cargo:rerun-if-changed={}", archive.display());
            println!("cargo:rerun-if-changed={}", manifest.display());
            fs::copy(&archive, out_dir.join("runtime-archive.payload"))
                .expect("copy embedded AOT runtime payload");
            fs::copy(&manifest, out_dir.join("runtime-archive.json"))
                .expect("copy embedded AOT runtime manifest");
            "pub static PAYLOAD: Option<&[u8]> = Some(include_bytes!(concat!(env!(\"OUT_DIR\"), \"/runtime-archive.payload\")));\npub static MANIFEST: Option<&str> = Some(include_str!(concat!(env!(\"OUT_DIR\"), \"/runtime-archive.json\")));\n".to_string()
        }
        _ => panic!(
            "RUNMAT_AOT_RUNTIME_ARCHIVE and RUNMAT_AOT_RUNTIME_MANIFEST must be set together"
        ),
    };
    fs::write(out_dir.join("embedded.rs"), generated).expect("write embedded AOT source");
}
