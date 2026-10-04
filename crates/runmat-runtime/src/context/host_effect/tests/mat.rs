use super::*;

fn workload<'a>(
    source: &'a Path,
    names: &'a [String],
    patterns: &'a [String],
    output: MatDecodeOutput,
) -> HostEffectWorkload<'a> {
    HostEffectWorkload::MatDecode {
        source,
        selection: MatDecodeSelection::Selected { names, patterns },
        output,
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mat_identity_separates_exact_path_selectors_and_publication_mode() {
    let names = vec!["A".to_owned()];
    let other = vec!["B".to_owned()];
    let patterns = vec!["^sensor_".to_owned()];
    let first = workload(
        Path::new("/data/one.mat"),
        &names,
        &patterns,
        MatDecodeOutput::Struct,
    )
    .identity_bytes();
    assert_ne!(
        first,
        workload(
            Path::new("/data/two.mat"),
            &names,
            &patterns,
            MatDecodeOutput::Struct
        )
        .identity_bytes()
    );
    assert_ne!(
        first,
        workload(
            Path::new("/data/one.mat"),
            &other,
            &patterns,
            MatDecodeOutput::Struct
        )
        .identity_bytes()
    );
    assert_ne!(
        first,
        workload(
            Path::new("/data/one.mat"),
            &names,
            &patterns,
            MatDecodeOutput::Workspace
        )
        .identity_bytes()
    );
    assert_eq!(
        HostEffectWorkload::MatDecode {
            source: Path::new("/data/one.mat"),
            selection: MatDecodeSelection::All,
            output: MatDecodeOutput::Workspace,
        }
        .output(),
        HostEffectOutput::EffectOnly
    );
}

#[cfg(unix)]
#[test]
fn mat_identity_preserves_non_utf8_path_bytes() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;
    let source = |suffix| {
        let mut raw = b"/mat-".to_vec();
        raw.push(suffix);
        PathBuf::from(OsString::from_vec(raw))
    };
    let first = source(0xff);
    let second = source(0xfe);
    assert_eq!(first.to_string_lossy(), second.to_string_lossy());
    let identity = |path: &Path| {
        HostEffectWorkload::MatDecode {
            source: path,
            selection: MatDecodeSelection::All,
            output: MatDecodeOutput::Struct,
        }
        .identity_bytes()
    };
    assert_ne!(identity(&first), identity(&second));
}
