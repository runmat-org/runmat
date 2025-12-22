//! Integration tests for runmat-ffi using the native mymath library.

use runmat_ffi::NativeLibrary;
use std::path::PathBuf;

fn get_test_lib_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("native");
    #[cfg(target_os = "windows")]
    path.push("mymath.dll");
    #[cfg(target_os = "linux")]
    path.push("libmymath.so");
    #[cfg(target_os = "macos")]
    path.push("libmymath.dylib");
    path
}

#[test]
fn test_load_library() {
    let path = get_test_lib_path();
    if !path.exists() {
        eprintln!("Test library not found at {:?}, skipping test", path);
        return;
    }

    let lib = NativeLibrary::load(&path).expect("Failed to load mymath library");
    assert!(lib.path().contains("mymath"));
}

#[test]
fn test_call_add() {
    let path = get_test_lib_path();
    if !path.exists() {
        eprintln!("Test library not found at {:?}, skipping test", path);
        return;
    }

    let lib = NativeLibrary::load(&path).expect("Failed to load mymath library");

    type AddFn = unsafe extern "C" fn(f64, f64) -> f64;
    let add: libloading::Symbol<AddFn> = unsafe { lib.get_function("add").unwrap() };

    let result = unsafe { add(2.0, 3.0) };
    assert!((result - 5.0).abs() < 1e-10);
}

#[test]
fn test_call_square() {
    let path = get_test_lib_path();
    if !path.exists() {
        eprintln!("Test library not found at {:?}, skipping test", path);
        return;
    }

    let lib = NativeLibrary::load(&path).expect("Failed to load mymath library");

    type SquareFn = unsafe extern "C" fn(f64) -> f64;
    let square: libloading::Symbol<SquareFn> = unsafe { lib.get_function("square").unwrap() };

    let result = unsafe { square(4.0) };
    assert!((result - 16.0).abs() < 1e-10);
}

#[test]
fn test_call_get_pi() {
    let path = get_test_lib_path();
    if !path.exists() {
        eprintln!("Test library not found at {:?}, skipping test", path);
        return;
    }

    let lib = NativeLibrary::load(&path).expect("Failed to load mymath library");

    type GetPiFn = unsafe extern "C" fn() -> f64;
    let get_pi: libloading::Symbol<GetPiFn> = unsafe { lib.get_function("get_pi").unwrap() };

    let result = unsafe { get_pi() };
    assert!((result - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn test_call_sum5() {
    let path = get_test_lib_path();
    if !path.exists() {
        eprintln!("Test library not found at {:?}, skipping test", path);
        return;
    }

    let lib = NativeLibrary::load(&path).expect("Failed to load mymath library");

    type Sum5Fn = unsafe extern "C" fn(f64, f64, f64, f64, f64) -> f64;
    let sum5: libloading::Symbol<Sum5Fn> = unsafe { lib.get_function("sum5").unwrap() };

    let result = unsafe { sum5(1.0, 2.0, 3.0, 4.0, 5.0) };
    assert!((result - 15.0).abs() < 1e-10);
}
