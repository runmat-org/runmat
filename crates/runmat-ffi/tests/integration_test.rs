//! Integration tests for runmat-ffi using the native mymath library.

use runmat_ffi::{NativeLibrary, SignatureFile};
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

fn get_test_sig_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("native");
    path.push("mymath.ffi");
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

// ============================================================================
// Signature file tests
// ============================================================================

#[test]
fn test_parse_signature_file() {
    let path = get_test_sig_path();
    if !path.exists() {
        eprintln!("Signature file not found at {:?}, skipping test", path);
        return;
    }

    let sigs = SignatureFile::parse_file(&path).expect("Failed to parse signature file");

    // Check that all expected functions are present
    assert!(sigs.contains("add"), "Should have 'add' signature");
    assert!(sigs.contains("square"), "Should have 'square' signature");
    assert!(sigs.contains("get_pi"), "Should have 'get_pi' signature");
    assert!(sigs.contains("sum5"), "Should have 'sum5' signature");
}

#[test]
fn test_signature_arg_counts() {
    let path = get_test_sig_path();
    if !path.exists() {
        return;
    }

    let sigs = SignatureFile::parse_file(&path).expect("Failed to parse signature file");

    // Check argument counts
    let add = sigs.get("add").unwrap();
    assert_eq!(add.args.len(), 2, "add should have 2 args");

    let square = sigs.get("square").unwrap();
    assert_eq!(square.args.len(), 1, "square should have 1 arg");

    let get_pi = sigs.get("get_pi").unwrap();
    assert_eq!(get_pi.args.len(), 0, "get_pi should have 0 args");

    let sum5 = sigs.get("sum5").unwrap();
    assert_eq!(sum5.args.len(), 5, "sum5 should have 5 args");
}

#[test]
fn test_signature_types() {
    let path = get_test_sig_path();
    if !path.exists() {
        return;
    }

    let sigs = SignatureFile::parse_file(&path).expect("Failed to parse signature file");

    // All functions should have f64 return type
    for sig in sigs.iter() {
        assert_eq!(
            sig.ret,
            runmat_ffi::FfiType::F64,
            "Function '{}' should return f64",
            sig.name
        );
    }

    // Check is_all_f64_scalar for scalar functions
    let add = sigs.get("add").unwrap();
    assert!(add.is_all_f64_scalar(), "add should be all f64 scalar");
}
