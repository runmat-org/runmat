use super::*;
use futures::executor::block_on;
use std::env;
use std::path::Path;

#[test]
fn source_admission_precedes_resolution_and_provider_access() {
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let error = block_on(evaluate(&[Value::from("missing-source"), resident])).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:pcode:InvalidArgument"));
    let pages = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap(),
    );
    assert_eq!(
        block_on(evaluate(&[pages])).unwrap_err().identifier(),
        Some("RunMat:pcode:InvalidArgument")
    );
    let row = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1])
            .unwrap(),
    );
    assert_eq!(
        parse_options(crate::call::FunctionArgumentView::from_slice(&[row]))
            .unwrap()
            .items,
        vec!["ab"]
    );
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    assert_eq!(
        block_on(evaluate(&[])).unwrap_err().identifier(),
        Some("RunMat:pcode:TooManyOutputs")
    );
}

#[test]
fn both_container_algorithms_preserve_source_bytes_and_reject_modified_payload() {
    for algorithm in [PcodeAlgorithm::R2007b, PcodeAlgorithm::R2022a] {
        let source = "% Unicode λ\nanswer = uint64(17);\n";
        let encoded = encode_pcode_source(source, "worker.m", algorithm);
        assert_eq!(decode_pcode_bytes(encoded.as_bytes()).unwrap(), source);
        let changed = encoded.replace("sha256:", "sha256:00");
        assert!(decode_pcode_bytes(changed.as_bytes()).is_err());
    }
}

struct CwdGuard {
    original: PathBuf,
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original);
    }
}

fn push_cwd(path: &Path) -> CwdGuard {
    let original = env::current_dir().expect("current dir");
    env::set_current_dir(path).expect("set current dir");
    CwdGuard { original }
}

#[test]
fn pcode_creates_obscured_executable_p_file_from_source_name() {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(temp.path().join("worker.m"), "answer = 42;\n").expect("write source");
    let _cwd = push_cwd(temp.path());

    let result = block_on(evaluate(&[Value::from("worker")])).expect("pcode");
    assert_eq!(result.generated.len(), 1);
    assert!(result.generated[0].ends_with("worker.p"));
    let encoded = std::fs::read_to_string(temp.path().join("worker.p")).expect("read pcode");
    assert!(encoded.starts_with(RUNMAT_PCODE_MAGIC));
    assert!(!encoded.contains("answer = 42"));
    let decoded = decode_pcode_bytes(encoded.as_bytes()).expect("decode pcode");
    assert_eq!(decoded, "answer = 42;\n");
}

#[test]
fn pcode_inplace_writes_next_to_source() {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let src_dir = temp.path().join("src");
    std::fs::create_dir_all(&src_dir).expect("create src");
    let source = src_dir.join("worker.m");
    std::fs::write(&source, "value = 7;\n").expect("write source");
    let _cwd = push_cwd(temp.path());

    let result = block_on(evaluate(&[
        Value::from(source.to_string_lossy().to_string()),
        Value::from("-inplace"),
    ]))
    .expect("pcode");
    assert_eq!(result.generated.len(), 1);
    assert!(result.generated[0].ends_with("src/worker.p"));
    assert!(src_dir.join("worker.p").is_file());
    assert!(!temp.path().join("worker.p").exists());
}

#[test]
fn pcode_folder_preserves_package_and_class_suffix_in_current_dir() {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    let source_root = temp.path().join("source").join("+pkg").join("@Widget");
    std::fs::create_dir_all(&source_root).expect("create source root");
    std::fs::write(source_root.join("Widget.m"), "classdef Widget\nend\n")
        .expect("write class source");
    let out = temp.path().join("out");
    std::fs::create_dir_all(&out).expect("create out");
    let _cwd = push_cwd(&out);

    let result = block_on(evaluate(&[Value::from(
        source_root.to_string_lossy().to_string(),
    )]))
    .expect("pcode");
    assert_eq!(result.generated.len(), 1);
    assert!(result.generated[0].ends_with("+pkg/@Widget/Widget.p"));
    assert!(out.join("+pkg/@Widget/Widget.p").is_file());
}

#[test]
fn pcode_wildcard_ignores_non_m_files() {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let temp = tempfile::TempDir::new().expect("tempdir");
    std::fs::write(temp.path().join("one.m"), "one = 1;\n").expect("write one");
    std::fs::write(temp.path().join("one.txt"), "not matlab\n").expect("write txt");
    let _cwd = push_cwd(temp.path());

    let result = block_on(evaluate(&[Value::from("one.*")])).expect("pcode");
    assert_eq!(result.generated.len(), 1);
    assert!(result.generated[0].ends_with("one.p"));
    assert!(temp.path().join("one.p").is_file());
    assert!(!temp.path().join("one.txt.p").exists());
}

#[test]
fn pcode_rejects_live_scripts_and_unknown_options() {
    let err = block_on(evaluate(&[Value::from("demo.mlx")]))
        .expect_err("live script source is unsupported");
    assert_eq!(err.identifier(), Some("RunMat:pcode:UnsupportedExtension"));

    let err = block_on(evaluate(&[Value::from("demo"), Value::from("-bad")]))
        .expect_err("unknown option is invalid");
    assert_eq!(err.identifier(), Some("RunMat:pcode:InvalidOption"));
}

#[test]
fn invalid_pcode_decode_reports_error() {
    let err = decode_pcode_bytes(b"not matlab pcode").expect_err("invalid pcode");
    assert!(err.contains("not a RunMat-generated"));
}
