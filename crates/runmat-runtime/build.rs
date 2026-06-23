use std::{
    env, fs,
    path::{Path, PathBuf},
};

const WASM_REGISTRY_ENV: &str = "RUNMAT_GENERATE_WASM_REGISTRY";
const WASM_REGISTRY_OUT_ENV: &str = "RUNMAT_WASM_REGISTRY_OUT";
const WASM_REGISTRY_RELATIVE_PATH: &str = "src/builtins/generated_wasm_registry.rs";
const WASM_REGISTRY_HELPER_MARKER: &str = "__runmat_wasm_register_builtin_";
const WASM_REGISTRY_ANY_HELPER_MARKER: &str = "__runmat_wasm_register_";

fn split_list(value: &str) -> Vec<String> {
    value
        .split([';', ':', ',', ' '])
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}

fn main() {
    ensure_wasm_registry_state();

    // Only act when BLAS/LAPACK feature is enabled
    if env::var("CARGO_FEATURE_BLAS_LAPACK").is_err() {
        return;
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // macOS links Accelerate via #[link] in lib.rs
    if target_os == "macos" {
        return;
    }

    // Help the linker find and link LAPACK/BLAS on Windows and Linux when using system libs.
    // We honor standard envs often used by lapack-sys/blas-sys and vcpkg.

    // vcpkg paths
    if let Ok(vcpkg_root) = env::var("VCPKG_ROOT") {
        let triplet = env::var("VCPKGRS_TRIPLET")
            .ok()
            .or_else(|| env::var("VCPKG_DEFAULT_TRIPLET").ok())
            .unwrap_or_else(|| "x64-windows".to_string());
        let lib_path = format!("{vcpkg_root}/installed/{triplet}/lib");
        println!("cargo:rustc-link-search=native={lib_path}");
        println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
        println!("cargo:rerun-if-env-changed=VCPKGRS_TRIPLET");
        println!("cargo:rerun-if-env-changed=VCPKG_DEFAULT_TRIPLET");
    }

    // OPENBLAS_DIR typically points at vcpkg installed prefix
    if let Ok(openblas_dir) = env::var("OPENBLAS_DIR") {
        println!("cargo:rustc-link-search=native={openblas_dir}/lib");
        println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
    }

    // Explicit LAPACK hints
    if let Ok(lapack_lib_dir) = env::var("LAPACK_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lapack_lib_dir}");
        println!("cargo:rerun-if-env-changed=LAPACK_LIB_DIR");
    }
    if let Ok(lapack_libs) = env::var("LAPACK_LIBS") {
        for lib in split_list(&lapack_libs) {
            println!("cargo:rustc-link-lib={lib}");
        }
        println!("cargo:rerun-if-env-changed=LAPACK_LIBS");
    } else {
        if target_os == "windows" {
            // vcpkg's OpenBLAS package supplies the LAPACK symbols we use.
            println!("cargo:rustc-link-lib=openblas");
        }
    }

    // Explicit BLAS hints
    if let Ok(blas_lib_dir) = env::var("BLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={blas_lib_dir}");
        println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");
    }
    if let Ok(blas_libs) = env::var("BLAS_LIBS") {
        for lib in split_list(&blas_libs) {
            println!("cargo:rustc-link-lib={lib}");
        }
        println!("cargo:rerun-if-env-changed=BLAS_LIBS");
    } else {
        // Fallback for typical setups
        println!("cargo:rustc-link-lib=openblas");
    }
}

/// Ensure the generated wasm registry is valid for wasm builds.
///
/// ORDERING CONSTRAINT: build.rs runs before this crate is compiled. The supported
/// regeneration flow writes a stub to a temporary file, lets the `#[runtime_builtin]`
/// proc-macros append helper calls while this crate compiles for wasm, then atomically
/// replaces the checked-in registry only after cargo succeeds. Normal wasm builds
/// validate that completed generated source rather than rewriting it.
///
/// Do NOT emit cargo:rerun-if-changed for the generated registry path. Proc-macros
/// modify that file during generation; adding rerun-if-changed would trigger build.rs
/// to re-run after every proc-macro write during generation, forcing repeated
/// runmat-runtime recompiles.
fn ensure_wasm_registry_state() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let generating = matches!(env::var(WASM_REGISTRY_ENV), Ok(value) if value == "1");
    println!("cargo:rerun-if-env-changed={WASM_REGISTRY_ENV}");
    println!("cargo:rerun-if-env-changed={WASM_REGISTRY_OUT_ENV}");

    if target_arch != "wasm32" {
        if generating {
            panic!("RunMat wasm builtin registry generation must target wasm32-unknown-unknown");
        }
        return;
    }

    let fingerprint_inputs = wasm_registry_fingerprint_inputs(&manifest_dir);
    for (_, input) in &fingerprint_inputs {
        println!("cargo:rerun-if-changed={}", input.display());
    }
    let fingerprint = wasm_registry_fingerprint(&fingerprint_inputs);
    let build_configuration = wasm_registry_build_configuration();
    let registry_path = manifest_dir.join(WASM_REGISTRY_RELATIVE_PATH);

    if generating {
        let out_path = env::var(WASM_REGISTRY_OUT_ENV).unwrap_or_else(|_| {
            panic!(
                "{WASM_REGISTRY_ENV}=1 requires {WASM_REGISTRY_OUT_ENV}; use \
scripts/regenerate-wasm-registry.sh so partial generation never overwrites the checked-in registry"
            )
        });
        let path = PathBuf::from(out_path);
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(
            &path,
            wasm_registry_stub(&fingerprint, &build_configuration),
        )
        .expect("failed to initialize wasm registry file");
    } else {
        println!("cargo:rerun-if-changed={}", registry_path.display());
        let registry = fs::read_to_string(&registry_path).unwrap_or_default();
        let errors = validate_wasm_registry(&registry, &fingerprint, &build_configuration);
        if !errors.is_empty() {
            panic!(
                "RunMat wasm builtin registry is missing, incomplete, stale, or was generated \
for a different wasm runtime configuration:\n- {}\nRun scripts/regenerate-wasm-registry.sh \
before building runmat-wasm.",
                errors.join("\n- ")
            );
        }
    }
}

fn wasm_registry_stub(fingerprint: &str, build_configuration: &str) -> String {
    format!(
        "// @generated by `scripts/regenerate-wasm-registry.sh`\n\
         pub const REGISTRY_COMPLETE: bool = false;\n\
         pub const REGISTRY_SOURCE_FINGERPRINT: &str = \"{fingerprint}\";\n\n\
         pub const REGISTRY_BUILD_CONFIGURATION: &str = \"{build_configuration}\";\n\
         pub const REGISTRY_ENTRY_COUNT: usize = 0;\n\n\
         pub fn register_all() {{\n}}\n"
    )
}

fn validate_wasm_registry(
    registry: &str,
    fingerprint: &str,
    build_configuration: &str,
) -> Vec<String> {
    let mut errors = Vec::new();
    let expected_fingerprint =
        format!("pub const REGISTRY_SOURCE_FINGERPRINT: &str = \"{fingerprint}\";");
    if !registry.contains(&expected_fingerprint) {
        errors.push("source fingerprint does not match current builtin sources".to_string());
    }

    let expected_configuration =
        format!("pub const REGISTRY_BUILD_CONFIGURATION: &str = \"{build_configuration}\";");
    if !registry.contains(&expected_configuration) {
        errors.push(format!(
            "build configuration does not match current target/features ({build_configuration})"
        ));
    }

    if !registry.contains("pub const REGISTRY_COMPLETE: bool = true;") {
        errors
            .push("registry was not marked complete by the atomic regeneration script".to_string());
    }

    let entry_count = registry.matches(WASM_REGISTRY_ANY_HELPER_MARKER).count();
    let builtin_count = registry.matches(WASM_REGISTRY_HELPER_MARKER).count();
    if entry_count == 0 || builtin_count == 0 {
        errors.push("registry contains no builtin helper registrations".to_string());
    }

    let expected_entry_count = format!("pub const REGISTRY_ENTRY_COUNT: usize = {entry_count};");
    if !registry.contains(&expected_entry_count) {
        errors.push(format!(
            "registry entry count metadata does not match generated helper calls ({entry_count})"
        ));
    }

    errors
}

fn wasm_registry_build_configuration() -> String {
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_string());
    let mut features = env::vars()
        .filter_map(|(key, _)| {
            key.strip_prefix("CARGO_FEATURE_")
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort();
    format!("target={target};features={}", features.join(","))
}

fn wasm_registry_fingerprint_inputs(manifest_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut inputs = Vec::new();
    collect_registry_source_files(
        &manifest_dir.join("src").join("builtins"),
        Path::new("src/builtins"),
        &mut inputs,
    );
    inputs.push((
        "../runmat-macros/src/lib.rs".to_string(),
        manifest_dir
            .parent()
            .expect("runtime crate should live under crates/")
            .join("runmat-macros")
            .join("src")
            .join("lib.rs"),
    ));
    inputs.sort_by(|left, right| left.0.cmp(&right.0));
    inputs
}

fn collect_registry_source_files(
    dir: &Path,
    label_dir: &Path,
    inputs: &mut Vec<(String, PathBuf)>,
) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", dir.display()))
        .map(|entry| entry.expect("failed to read wasm registry source entry"))
        .collect();
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        let label = label_dir.join(entry.file_name());
        if path.is_dir() {
            collect_registry_source_files(&path, &label, inputs);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) == Some("generated_wasm_registry.rs") {
            continue;
        }
        inputs.push((label.to_string_lossy().replace('\\', "/"), path));
    }
}

fn wasm_registry_fingerprint(inputs: &[(String, PathBuf)]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for (label, path) in inputs {
        update_registry_hash(&mut hash, label.as_bytes());
        update_registry_hash(&mut hash, &[0]);
        let bytes =
            fs::read(path).unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        update_registry_hash(&mut hash, &bytes);
        update_registry_hash(&mut hash, &[0xff]);
    }
    format!("fnv1a64-{hash:016x}")
}

fn update_registry_hash(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}
