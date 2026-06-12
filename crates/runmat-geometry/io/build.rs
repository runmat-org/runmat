#[cfg(feature = "occt-native")]
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_ROOT");
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=RUNMAT_OCCT_LINK_MODE");
    println!("cargo:rerun-if-env-changed=CMAKE_POLICY_VERSION_MINIMUM");

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
    occt_sys::build_occt();
    let occt_path = occt_sys::occt_path();
    (occt_path.join("include"), occt_path.join("lib"))
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
