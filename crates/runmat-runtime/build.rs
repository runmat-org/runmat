use std::{env, fs, path::PathBuf};

fn split_list(value: &str) -> Vec<String> {
    value
        .split([';', ':', ',', ' '])
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}

fn main() {
    ensure_wasm_registry_stub();

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
        // Reasonable defaults on Windows with vcpkg
        if target_os == "windows" {
            // Try lapack first, then openblas
            println!("cargo:rustc-link-lib=lapack");
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

/// Ensure the generated wasm registry file exists so include! does not fail in wasm builds.
/// The proc-macro will overwrite/extend this file when RUNMAT_GENERATE_WASM_REGISTRY=1;
/// this stub keeps the compile happy when the generator has not yet run.
///
/// ORDERING CONSTRAINT: build.rs runs *after* all dependency crates finish compiling (cargo
/// compiles deps before running the dependent's build script). The `#[runtime_builtin]`
/// proc-macros in runmat-macros populate this file during runmat-builtins compilation.
/// Therefore this function must NOT reset the file — doing so overwrites proc-macro output
/// and leaves include!() reading an empty registry.
///
/// Do NOT emit cargo:rerun-if-changed for this file. Proc-macros modify it during builds;
/// adding rerun-if-changed would trigger build.rs to re-run after every proc-macro write,
/// which resets the file, which forces a recompile of runmat-runtime, which re-runs
/// proc-macros — an infinite rebuild loop that manifests as cargo stuck at the wasm test
/// targets indefinitely.
fn ensure_wasm_registry_stub() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../target/runmat_wasm_registry.rs");
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    // Only create a placeholder when no file exists yet. Once proc-macros have run they
    // own the file content; build.rs must not touch it after that point.
    if !path.exists() {
        let _ = fs::write(&path, "pub fn register_all() {\n}\n");
    }
    // Re-run only when the env var itself changes (switching between generation and normal
    // build modes). Do NOT add rerun-if-changed for the registry path — see above.
    println!("cargo:rerun-if-env-changed=RUNMAT_GENERATE_WASM_REGISTRY");
}
