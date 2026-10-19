#[path = "build_support/meshing_fingerprint.rs"]
mod meshing_fingerprint;
fn main() {
    let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest.parent().unwrap().parent().unwrap();
    let mut environment = std::env::vars()
        .filter(|(key, _)| {
            key.starts_with("CARGO_FEATURE_")
                || key.starts_with("CARGO_CFG_")
                || matches!(
                    key.as_str(),
                    "TARGET"
                        | "HOST"
                        | "OPT_LEVEL"
                        | "DEBUG"
                        | "PROFILE"
                        | "CARGO_ENCODED_RUSTFLAGS"
                )
        })
        .collect::<Vec<_>>();
    environment.sort();
    let compiler = std::process::Command::new(std::env::var_os("RUSTC").unwrap())
        .arg("--version")
        .output()
        .unwrap();
    assert!(compiler.status.success(), "compiler identity unavailable");
    let context = format!(
        "{environment:?}\n{}",
        String::from_utf8_lossy(&compiler.stdout)
    );
    let metadata = meshing_fingerprint::metadata(workspace).unwrap_or_else(|error| {
        println!("cargo:warning=Meshing cache dependency closure unavailable: {error}");
        None
    });
    for (name, volume) in [
        ("RUNMAT_MESHING_UPSTREAM_IMPLEMENTATION", false),
        ("RUNMAT_MESHING_VOLUME_IMPLEMENTATION", true),
    ] {
        let identity = meshing_fingerprint::fingerprint(
            workspace,
            volume,
            context.as_bytes(),
            true,
            metadata.as_ref(),
        )
        .unwrap_or_else(|error| panic!("meshing implementation inventory failed: {error}"));
        println!("cargo:rustc-env={name}={identity}");
    }
}
