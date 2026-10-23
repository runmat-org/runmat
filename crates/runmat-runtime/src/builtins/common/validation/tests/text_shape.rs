use super::*;

fn shaped_text(text: &str, pages: bool) -> Value {
    let data = text.chars().collect::<Vec<_>>();
    let unit_len = text.encode_utf16().count();
    let shape = if pages {
        vec![1, 1, unit_len]
    } else {
        vec![1, unit_len, 1]
    };
    Value::CharArray(CharArray::from_unicode_scalars_with_shape(data, shape).unwrap())
}

#[test]
fn text_admission_ignores_only_trailing_singletons() {
    let row = shaped_text("valid_name", false);
    let pages = shaped_text("valid_name", true);
    for name in [
        "mustBeText",
        "mustBeTextScalar",
        "mustBeNonzeroLengthText",
        "mustBeValidVariableName",
    ] {
        ok(name, vec![row.clone()]);
        err(name, vec![pages.clone()]);
    }
    assert!(isvarname_value(&row));
    assert!(!isvarname_value(&pages));
    for name in ["mustBeA", "mustBeUnderlyingType"] {
        ok(name, vec![Value::Num(1.0), shaped_text("double", false)]);
        let error =
            invoke_validator(name, vec![Value::Num(1.0), shaped_text("double", true)]).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:argumentValidation:InvalidArgument")
        );
    }
}

#[test]
fn remaining_file_and_folder_consumers_share_row_admission() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("validator-shape.txt");
    std::fs::write(&file, b"fixture").unwrap();
    for (name, path) in [
        ("mustBeFile", file.as_path()),
        ("mustBeFolder", directory.path()),
    ] {
        let text = path.to_str().unwrap();
        ok(name, vec![shaped_text(text, false)]);
        let error = invoke_validator(name, vec![shaped_text(text, true)]).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:argumentValidation:InvalidArgument")
        );
    }
}
