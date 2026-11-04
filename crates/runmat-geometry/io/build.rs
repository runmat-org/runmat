#[cfg(feature = "occt-native")]
use std::{env, path::PathBuf};

fn main() {
    runmat_build_identity::emit_configuration();
    // runmat-cad's integration tests, built into tests/cad_kernel_corpus.rs, run on the OCCT
    // kernel under test when this cfg is set (runmat-cad/tests/common/mod.rs).
    println!("cargo:rustc-check-cfg=cfg(runmat_cad_kernel_under_test)");
    println!("cargo:rustc-cfg=runmat_cad_kernel_under_test");
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
    let (occt_include, occt_lib, uses_bundled_occt) = occt_paths(&link_mode);

    let occt_link_libs = occt_link_libs(&occt_lib);
    let k1 = env::var_os("CARGO_FEATURE_OCCT_KERNEL").is_some();
    let mut bridges = vec!["src/occt/ffi.rs"];
    if k1 {
        bridges.push("src/occt/kernel_ffi.rs");
    }
    let mut build = cxx_build::bridges(bridges);
    build
        .file("src/occt/occt_bridge.cc")
        .file("src/occt/exact_assembly.cc")
        .file("src/occt/exact_identity.cc")
        .file("src/occt/exact_healing.cc")
        .file("src/occt/exact_small_topology.cc")
        .file("src/occt/exact_xcaf_healing.cc")
        .file("src/occt/exact_xcaf_subshape_remap.cc")
        .file("src/occt/exact_nesting.cc")
        .file("src/occt/exact_names.cc")
        .file("src/occt/exact_shared_interfaces.cc")
        .file("src/occt/exact_topology.cc")
        .file("src/occt/exact_wire.cc")
        .file("src/occt/exact_evaluator.cc")
        .file("src/occt/exact_feature.cc")
        .file("src/occt/exact_feature_operations.cc")
        .file("src/occt/exact_surface_evaluator.cc")
        .include(".")
        .include(&occt_include)
        .std("c++17")
        .flag_if_supported("-Wno-deprecated-declarations")
        .define("_USE_MATH_DEFINES", None);
    if k1 {
        // K1: the geometry-engine interface (kernel programs, profiles, tools, analyses,
        // hidden lines, AP242 export).
        for f in [
            "src/occt/exact_kernel.cc",
            "src/occt/exact_sketch_profile.cc",
            "src/occt/exact_sweep_loft.cc",
            "src/occt/exact_hlr.cc",
            "src/occt/exact_step_writer.cc",
        ] {
            build.file(f);
        }
    }
    if uses_bundled_occt || (link_mode == "static" && is_occt_78(&occt_include)) {
        // OCCT 7.8's process-lifetime RTTI registry can be destroyed before STEP/IGES type
        // descriptors when statically linked. This source overrides that one TKernel object with
        // the upstream lifetime-safe ownership model used by later OCCT releases.
        build.file("src/occt/occt_type_lifetime.cc");
    }

    if target.contains("emscripten") && is_occt_78(&occt_include) {
        // Override the two affected SDK archive members using upstream OCCT
        // implementation without native signal-to-longjmp conversion. The
        // bridge archive precedes OCCT, as for the RTTI lifetime adapter above.
        build.file("src/occt/wasm_shape_upgrade_shape_divide.cc");
        build.file("src/occt/wasm_shape_custom_bspline_restriction.cc");
    }

    if is_windows_gnu {
        build.define("OCC_CONVERT_SIGNALS", "TRUE");
    }
    if target_env == "msvc" {
        build.flag("/utf-8");
        // CXX gates shared-field defaults on the reported language standard.
        // MSVC otherwise reports pre-C++11 even with /std:c++17 selected.
        build.flag("/Zc:__cplusplus");
        // Preserve RAII cleanup when the bridge catches C++ exceptions.
        build.flag("/EHsc");
    }

    build.compile("runmat_geometry_io_occt");

    // Keep the bridge archive before OCCT's archives so its bundled-7.8 Standard_Type lifetime
    // override satisfies TKernel's symbols before the defective object can be selected.
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

    println!("cargo:rerun-if-changed=src/occt/ffi.rs");
    for k1 in [
        "src/occt/kernel_ffi.rs",
        "src/occt/exact_kernel.hxx",
        "src/occt/exact_kernel_ffi.hxx",
        "src/occt/exact_kernel.cc",
        "src/occt/exact_sketch_profile.cc",
        "src/occt/exact_sweep_loft.cc",
        "src/occt/exact_hlr.cc",
        "src/occt/exact_step_writer.cc",
    ] {
        println!("cargo:rerun-if-changed={k1}");
    }
    println!("cargo:rerun-if-changed=src/occt/occt_bridge.hxx");
    println!("cargo:rerun-if-changed=src/occt/occt_bridge.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_names.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_names.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_shared_interfaces.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_shared_interfaces.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_assembly.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_assembly.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_identity.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_identity.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_healing.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_healing.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_small_topology.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_xcaf_healing.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_xcaf_healing.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_xcaf_subshape_remap.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_xcaf_subshape_remap.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_nesting.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_nesting.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_topology.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_topology.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_wire.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_wire.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_evaluator.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_feature.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_feature_internal.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_feature_operations.cc");
    println!("cargo:rerun-if-changed=src/occt/exact_evaluator_internal.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_surface_bounds.hxx");
    println!("cargo:rerun-if-changed=src/occt/exact_surface_evaluator.cc");
    println!("cargo:rerun-if-changed=src/occt/wasm_shape_upgrade_shape_divide.cc");
    println!("cargo:rerun-if-changed=src/occt/wasm_shape_custom_bspline_restriction.cc");
}

#[cfg(feature = "occt-native")]
fn is_occt_78(include: &std::path::Path) -> bool {
    let version_header = include.join("Standard_Version.hxx");
    println!("cargo:rerun-if-changed={}", version_header.display());
    let Ok(header) = std::fs::read_to_string(version_header) else {
        return false;
    };
    let macro_value = |name: &str| {
        header.lines().find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next() == Some("#define") && words.next() == Some(name))
                .then(|| words.next())
                .flatten()
        })
    };
    macro_value("OCC_VERSION_MAJOR") == Some("7") && macro_value("OCC_VERSION_MINOR") == Some("8")
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
fn occt_paths(link_mode: &str) -> (PathBuf, PathBuf, bool) {
    let include_dir = env::var_os("RUNMAT_OCCT_INCLUDE_DIR").map(PathBuf::from);
    let lib_dir = env::var_os("RUNMAT_OCCT_LIB_DIR").map(PathBuf::from);
    match (include_dir, lib_dir) {
        (Some(include), Some(lib)) => return (include, lib, false),
        (Some(_), None) | (None, Some(_)) => {
            panic!("RUNMAT_OCCT_INCLUDE_DIR and RUNMAT_OCCT_LIB_DIR must be set together")
        }
        (None, None) => {}
    }

    if let Some(root) = env::var_os("RUNMAT_OCCT_ROOT").map(PathBuf::from) {
        return (occt_root_include_dir(&root), root.join("lib"), false);
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
    (occt_path.join("include"), occt_path.join("lib"), true)
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
        Ok(output) if output.status.success() => return,
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
