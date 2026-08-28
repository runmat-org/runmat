use std::io::{Cursor, Write as _};

use super::*;

fn wheel(files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    {
        let mut archive = zip::ZipWriter::new(&mut output);
        for (name, bytes) in files {
            archive
                .start_file(*name, zip::write::SimpleFileOptions::default())
                .expect("start wheel file");
            archive.write_all(bytes).expect("write wheel file");
        }
        archive.finish().expect("finish wheel");
    }
    output.into_inner()
}

fn valid_entry(bytes: Vec<u8>) -> PythonArtifactBundleEntry {
    PythonArtifactBundleEntry::wheel("example", "example", "example-1-py3-none-any.whl", bytes)
}

#[test]
fn canonical_bundle_round_trips_and_materializes_import_roots() {
    let bytes = wheel(&[
        ("example/__init__.py", b"def answer():\n    return 42\n"),
        (
            "example-1.dist-info/METADATA",
            b"Name: example\nVersion: 1\n",
        ),
    ]);
    let bundle = PythonArtifactBundle::artifacts_only(vec![valid_entry(bytes)]).unwrap();
    let encoded = bundle.canonical_bytes().unwrap();
    assert!(
        encoded.len() < bundle.artifacts[0].bytes.len() + 1_024,
        "canonical framing must not expand wheel bytes into JSON integer arrays"
    );
    assert_eq!(
        PythonArtifactBundle::from_canonical_bytes(&encoded).unwrap(),
        bundle
    );
    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(PythonArtifactBundle::from_canonical_bytes(&trailing).is_err());
    assert!(PythonArtifactBundle::from_canonical_bytes(&encoded[..encoded.len() - 1]).is_err());
    let installed = bundle.install().unwrap();
    assert_eq!(installed.module_paths().len(), 1);
    assert!(installed.module_paths()[0]
        .join("example/__init__.py")
        .is_file());
    if let Ok(installation) = crate::discover_python(&crate::PythonDiscoveryRequest::default()) {
        let session = crate::PythonSession::start(crate::PythonSessionConfig {
            installation,
            module_paths: installed.module_paths().to_vec(),
        })
        .expect("start CPython with the wheel import root");
        let values = session
            .invoke(crate::PythonCall::InvokeQualified {
                name: "example.answer".into(),
                arguments: Vec::new(),
            })
            .expect("import and invoke the packaged module");
        assert!(matches!(
            values.as_slice(),
            [crate::PythonValue::Signed(42)]
        ));
    }
}

#[test]
fn wheel_library_scheme_is_relocated_to_the_import_root() {
    let bytes = wheel(&[
        ("example-1.data/purelib/example.py", b"VALUE = 42\n"),
        (
            "example-1.dist-info/METADATA",
            b"Name: example\nVersion: 1\n",
        ),
    ]);
    let bundle = PythonArtifactBundle::artifacts_only(vec![valid_entry(bytes)]).unwrap();
    let installed = bundle.install().unwrap();
    assert!(installed.module_paths()[0].join("example.py").is_file());
    assert!(!installed.module_paths()[0].join("example-1.data").exists());
}

#[test]
fn wheel_identity_and_archive_paths_are_enforced() {
    let bytes = wheel(&[
        ("../example.py", b"VALUE = 42\n"),
        (
            "example-1.dist-info/METADATA",
            b"Name: example\nVersion: 1\n",
        ),
    ]);
    let error = PythonArtifactBundle::artifacts_only(vec![valid_entry(bytes)]).unwrap_err();
    assert!(error.to_string().contains("unsafe path"));

    let bytes = wheel(&[
        ("example.py", b"VALUE = 42\n"),
        (
            "example-1.dist-info/METADATA",
            b"Name: example\nVersion: 1\n",
        ),
    ]);
    let mut entry = valid_entry(bytes);
    entry.bytes.push(0);
    let error = PythonArtifactBundle::artifacts_only(vec![entry]).unwrap_err();
    assert!(error.to_string().contains("does not match its identity"));

    let bytes = wheel(&[
        ("example.py", b"VALUE = 42\n"),
        (
            "example-1.dist-info/METADATA",
            b"Name: example\nVersion: 1\n",
        ),
    ]);
    let entry = PythonArtifactBundleEntry::wheel(
        "../escape",
        "example",
        "example-1-py3-none-any.whl",
        bytes,
    );
    let error = PythonArtifactBundle::artifacts_only(vec![entry]).unwrap_err();
    assert!(error.to_string().contains("logical name"));
}
