#[cfg(feature = "occt-native")]
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_ROOT");
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_LINK_MODE");
    println!("cargo:rerun-if-env-changed=CMAKE_POLICY_VERSION_MINIMUM");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rerun-if-env-changed=CMAKE_OSX_DEPLOYMENT_TARGET");

    if std::env::var_os("CARGO_FEATURE_OCCT_NATIVE").is_none() {
        return;
    }

    build_occt_backend();
}

#[cfg(feature = "occt-native")]
fn build_occt_backend() {
    let target = env::var("TARGET").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let is_windows = target.to_ascii_lowercase().contains("windows");
    let is_windows_gnu = target.to_ascii_lowercase().contains("windows-gnu");
    configure_macos_deployment_target(&target);

    let link_mode = env::var("RUNMAT_OCCT_LINK_MODE").unwrap_or_else(|_| "static".to_string());
    if link_mode != "static" && link_mode != "dylib" {
        panic!("RUNMAT_OCCT_LINK_MODE must be either 'static' or 'dylib'");
    }
    let (occt_include, occt_lib) = occt_paths(&link_mode);

    let occt_link_libs = occt_link_libs(&occt_lib);
    println!("cargo:rustc-link-search=native={}", occt_lib.display());
    for lib in &occt_link_libs {
        println!("cargo:rustc-link-lib={link_mode}={lib}");
    }

    if is_windows {
        println!("cargo:rustc-link-lib=dylib=user32");
    }
    if target.to_ascii_lowercase().contains("apple-darwin") {
        println!("cargo:rustc-link-lib=dylib=objc");
        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rustc-link-lib=framework=IOKit");
    }
    if link_mode == "dylib" && !is_windows {
        for rpath in occt_rpaths(&occt_lib) {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");
        }
    }

    let mut build = cxx_build::bridge("src/occt/ffi.rs");
    build
        .file("src/occt/occt_bridge.cc")
        .include(&occt_include)
        .std("c++17")
        .flag_if_supported("-Wno-deprecated-declarations")
        .define("_USE_MATH_DEFINES", None);

    if is_windows_gnu {
        build.define("OCC_CONVERT_SIGNALS", "TRUE");
    }
    if target_env == "msvc" {
        build.flag("/utf-8");
    }

    build.compile("runmat_geometry_io_occt");

    println!("cargo:rerun-if-changed=src/occt/ffi.rs");
    println!("cargo:rerun-if-changed=src/occt/occt_bridge.hxx");
    println!("cargo:rerun-if-changed=src/occt/occt_bridge.cc");
}

#[cfg(not(feature = "occt-native"))]
fn build_occt_backend() {}

#[cfg(feature = "occt-native")]
fn configure_macos_deployment_target(target: &str) {
    if !target.to_ascii_lowercase().contains("apple-darwin") {
        return;
    }

    let deployment_target =
        env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".to_string());
    env::set_var("MACOSX_DEPLOYMENT_TARGET", &deployment_target);
    if env::var_os("CMAKE_OSX_DEPLOYMENT_TARGET").is_none() {
        env::set_var("CMAKE_OSX_DEPLOYMENT_TARGET", deployment_target);
    }
}

#[cfg(feature = "occt-native")]
fn occt_paths(link_mode: &str) -> (PathBuf, PathBuf) {
    let include_dir = env::var_os("RUNMAT_OCCT_INCLUDE_DIR").map(PathBuf::from);
    let lib_dir = env::var_os("RUNMAT_OCCT_LIB_DIR").map(PathBuf::from);
    match (include_dir, lib_dir) {
        (Some(include), Some(lib)) => return (include, lib),
        (Some(_), None) | (None, Some(_)) => {
            panic!("RUNMAT_OCCT_INCLUDE_DIR and RUNMAT_OCCT_LIB_DIR must be set together")
        }
        (None, None) => {}
    }

    if let Some(root) = env::var_os("RUNMAT_OCCT_ROOT").map(PathBuf::from) {
        return (occt_root_include_dir(&root), root.join("lib"));
    }

    if link_mode == "dylib" {
        panic!(
            "RUNMAT_OCCT_LINK_MODE=dylib requires RUNMAT_OCCT_ROOT or \
             RUNMAT_OCCT_INCLUDE_DIR/RUNMAT_OCCT_LIB_DIR; the bundled occt-sys build produces static OCCT libraries"
        );
    }

    if env::var_os("CMAKE_POLICY_VERSION_MINIMUM").is_none() {
        env::set_var("CMAKE_POLICY_VERSION_MINIMUM", "3.5");
    }
    require_cmake_for_bundled_occt();
    occt_sys::build_occt();
    let occt_path = occt_sys::occt_path();
    (occt_path.join("include"), occt_path.join("lib"))
}

#[cfg(feature = "occt-native")]
fn require_cmake_for_bundled_occt() {
    let target = env::var("TARGET").unwrap_or_default();
    let cmake_env_keys = cmake_env_keys(&target);
    for key in &cmake_env_keys {
        println!("cargo:rerun-if-env-changed={key}");
    }

    let (source, cmake) = configured_cmake(&cmake_env_keys);
    match std::process::Command::new(&cmake).arg("--version").output() {
        Ok(output) if output.status.success() => {
            println!(
                "cargo:warning=RunMat OCCT: building bundled OCCT with CMake from {source}: {}",
                cmake.display()
            );
            return;
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let detail = first_non_empty_line(&stderr)
                .or_else(|| first_non_empty_line(&stdout))
                .unwrap_or("no diagnostic output");
            announce_missing_cmake(
                &source,
                &cmake,
                &cmake_env_keys,
                &format!("command exited with status {} ({detail})", output.status),
            );
        }
        Err(err) => {
            announce_missing_cmake(&source, &cmake, &cmake_env_keys, &err.to_string());
        }
    }

    panic!("missing CMake required for bundled OCCT (occt-native)");
}

#[cfg(feature = "occt-native")]
fn cmake_env_keys(target: &str) -> Vec<String> {
    let target_env_key = target.replace('-', "_");
    let mut keys = Vec::new();
    if !target.is_empty() {
        keys.push(format!("CMAKE_{target}"));
    }
    if !target_env_key.is_empty() && target_env_key != target {
        keys.push(format!("CMAKE_{target_env_key}"));
    }
    keys.push("HOST_CMAKE".to_string());
    keys.push("CMAKE".to_string());
    keys
}

#[cfg(feature = "occt-native")]
fn configured_cmake(cmake_env_keys: &[String]) -> (String, PathBuf) {
    for key in cmake_env_keys {
        if let Some(value) = env::var_os(key).filter(|value| !value.is_empty()) {
            return (key.clone(), PathBuf::from(value));
        }
    }
    ("PATH".to_string(), PathBuf::from("cmake"))
}

#[cfg(feature = "occt-native")]
fn first_non_empty_line(text: &str) -> Option<&str> {
    text.lines().map(str::trim).find(|line| !line.is_empty())
}

#[cfg(feature = "occt-native")]
fn announce_missing_cmake(
    source: &str,
    cmake: &std::path::Path,
    cmake_env_keys: &[String],
    detail: &str,
) {
    println!(
        "cargo:warning=RunMat OCCT: occt-native is enabled and no external OCCT installation was configured."
    );
    println!(
        "cargo:warning=RunMat OCCT: building bundled OCCT requires a working CMake executable."
    );
    println!(
        "cargo:warning=RunMat OCCT: checked {source} for `{}`: {detail}",
        cmake.display()
    );
    println!(
        "cargo:warning=RunMat OCCT: accepted CMake overrides: {}",
        cmake_env_keys.join(", ")
    );
    println!(
        "cargo:warning=RunMat OCCT: install CMake (macOS: `brew install cmake`), set CMAKE=/path/to/cmake, or set RUNMAT_OCCT_ROOT / RUNMAT_OCCT_INCLUDE_DIR+RUNMAT_OCCT_LIB_DIR."
    );
}

#[cfg(feature = "occt-native")]
fn occt_root_include_dir(root: &std::path::Path) -> PathBuf {
    let include = root.join("include");
    let opencascade_include = include.join("opencascade");
    if opencascade_include.exists() {
        opencascade_include
    } else {
        include
    }
}

#[cfg(feature = "occt-native")]
fn occt_rpaths(occt_lib: &std::path::Path) -> Vec<String> {
    let mut rpaths = vec![occt_lib.display().to_string()];
    if let Some(extra) = env::var_os("RUNMAT_OCCT_RPATHS") {
        rpaths.extend(env::split_paths(&extra).filter_map(|path| {
            let value = path.display().to_string();
            (!value.is_empty()).then_some(value)
        }));
    }
    rpaths.sort();
    rpaths.dedup();
    rpaths
}

#[cfg(feature = "occt-native")]
fn occt_link_libs(occt_lib: &std::path::Path) -> Vec<&'static str> {
    let mut libs = Vec::new();
    push_first_existing(
        &mut libs,
        occt_lib,
        &[
            &["TKDESTEP"],
            &["TKSTEP", "TKSTEPAttr", "TKSTEPBase", "TKSTEP209"],
        ],
        "STEP",
    );
    push_first_existing(&mut libs, occt_lib, &[&["TKDEIGES"], &["TKIGES"]], "IGES");
    push_existing(
        &mut libs,
        occt_lib,
        &[
            "TKDE",
            "TKXSBase",
            "TKXCAF",
            "TKVCAF",
            "TKV3d",
            "TKHLR",
            "TKService",
            "TKLCAF",
            "TKCAF",
            "TKCDF",
            "TKBRep",
            "TKMesh",
            "TKShHealing",
            "TKFillet",
            "TKBool",
            "TKBO",
            "TKOffset",
            "TKFeat",
            "TKPrim",
            "TKTopAlgo",
            "TKGeomAlgo",
            "TKGeomBase",
            "TKG3d",
            "TKG2d",
            "TKMath",
            "TKernel",
        ],
    );
    libs
}

#[cfg(feature = "occt-native")]
fn push_first_existing(
    libs: &mut Vec<&'static str>,
    occt_lib: &std::path::Path,
    groups: &[&[&'static str]],
    label: &str,
) {
    for group in groups {
        if group
            .iter()
            .any(|candidate| native_library_exists(occt_lib, candidate))
        {
            libs.extend(*group);
            return;
        }
    }
    panic!(
        "OCCT {label} libraries were not found in {}",
        occt_lib.display()
    );
}

#[cfg(feature = "occt-native")]
fn push_existing(
    libs: &mut Vec<&'static str>,
    occt_lib: &std::path::Path,
    candidates: &[&'static str],
) {
    libs.extend(
        candidates
            .iter()
            .copied()
            .filter(|candidate| native_library_exists(occt_lib, candidate)),
    );
}

#[cfg(feature = "occt-native")]
fn native_library_exists(occt_lib: &std::path::Path, lib: &str) -> bool {
    [
        format!("lib{lib}.a"),
        format!("{lib}.lib"),
        format!("lib{lib}.so"),
        format!("lib{lib}.dylib"),
    ]
    .iter()
    .any(|name| occt_lib.join(name).exists())
}
