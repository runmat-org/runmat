use std::env;

fn split_list(value: &str) -> Vec<String> {
    value
        .split(|c| c == ';' || c == ':' || c == ',' || c == ' ')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}

fn main() {
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
        let lib_path = format!("{}/installed/{}/lib", vcpkg_root, triplet);
        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
        println!("cargo:rerun-if-env-changed=VCPKGRS_TRIPLET");
        println!("cargo:rerun-if-env-changed=VCPKG_DEFAULT_TRIPLET");
    }

    // OPENBLAS_DIR typically points at vcpkg installed prefix
    if let Ok(openblas_dir) = env::var("OPENBLAS_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", openblas_dir);
        println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
    }

    // Explicit LAPACK hints
    if let Ok(lapack_lib_dir) = env::var("LAPACK_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lapack_lib_dir);
        println!("cargo:rerun-if-env-changed=LAPACK_LIB_DIR");
    }
    if let Ok(lapack_libs) = env::var("LAPACK_LIBS") {
        for lib in split_list(&lapack_libs) {
            println!("cargo:rustc-link-lib={}", lib);
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
        println!("cargo:rustc-link-search=native={}", blas_lib_dir);
        println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");
    }
    if let Ok(blas_libs) = env::var("BLAS_LIBS") {
        for lib in split_list(&blas_libs) {
            println!("cargo:rustc-link-lib={}", lib);
        }
        println!("cargo:rerun-if-env-changed=BLAS_LIBS");
    } else {
        // Fallback for typical setups
        println!("cargo:rustc-link-lib=openblas");
    }
}
