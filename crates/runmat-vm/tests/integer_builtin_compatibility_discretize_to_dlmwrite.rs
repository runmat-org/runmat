#[path = "support/mod.rs"]
mod test_helpers;

use runmat_value::{IntegerStorage, Value};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use test_helpers::execute_source;

const INTEGER_CONSTRUCTORS: [&str; 8] = [
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
];

fn expected_storage(constructor: &str, signed: &[i64], unsigned: &[u64]) -> IntegerStorage {
    match constructor {
        "int8" => IntegerStorage::I8(signed.iter().map(|&value| value as i8).collect()),
        "int16" => IntegerStorage::I16(signed.iter().map(|&value| value as i16).collect()),
        "int32" => IntegerStorage::I32(signed.iter().map(|&value| value as i32).collect()),
        "int64" => IntegerStorage::I64(signed.to_vec()),
        "uint8" => IntegerStorage::U8(unsigned.iter().map(|&value| value as u8).collect()),
        "uint16" => IntegerStorage::U16(unsigned.iter().map(|&value| value as u16).collect()),
        "uint32" => IntegerStorage::U32(unsigned.iter().map(|&value| value as u32).collect()),
        "uint64" => IntegerStorage::U64(unsigned.to_vec()),
        _ => unreachable!("known integer constructor"),
    }
}

fn contains_integer_storage(values: &[Value], expected: &IntegerStorage) -> bool {
    values.iter().any(
        |value| matches!(value, Value::Tensor(tensor) if tensor.integer_storage() == Some(expected)),
    )
}

fn contains_numeric_tensor(values: &[Value], shape: &[usize], expected: &[f64]) -> bool {
    values.iter().any(|value| {
        matches!(value, Value::Tensor(tensor)
            if tensor.shape == shape && tensor.materialize_f64() == expected)
    })
}

#[test]
fn compiled_discretize_and_downsample_preserve_every_integer_class_exactly() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            "bins = discretize([-1 0.5 1.5 3], [0 1 2], {constructor}([5 7])); sampled = downsample({constructor}([1 2 3 4 5]), 2, 1);"
        );
        let values = execute_source(&source).expect("compiled structural integer semantics");
        let bins = expected_storage(constructor, &[0, 5, 7, 0], &[0, 5, 7, 0]);
        let sampled = expected_storage(constructor, &[2, 4], &[2, 4]);
        assert!(
            contains_integer_storage(&values, &bins),
            "{constructor} bins"
        );
        assert!(
            contains_integer_storage(&values, &sampled),
            "{constructor} downsample"
        );
    }
}

#[test]
fn compiled_integer_controls_cover_dividerand_dot_doc2sequence_and_duration() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            r#"
            rng('default');
            [train, validation, test] = dividerand({constructor}(3), 1, 0, 0);
            total = numel(train) + numel(validation) + numel(test);
            products = dot([1 2; 3 4], [1 1; 1 1], {constructor}(1));
            enc = wordEncoding(["alpha" "beta"]);
            docs = tokenizedDocument("alpha");
            sequences = doc2sequence(enc, docs, "Length", {constructor}(2), "PaddingValue", {constructor}(7), "PaddingDirection", "right");
            sequence = sequences{{1}};
            elapsed = duration({constructor}([1 15 0]));
            elapsed_text = char(elapsed);
            "#
        );
        let values = execute_source(&source).expect("compiled integer controls");
        assert!(
            values
                .iter()
                .any(|value| matches!(value, Value::Num(value) if *value == 3.0)),
            "{constructor} dividerand total"
        );
        assert!(
            contains_numeric_tensor(&values, &[1, 2], &[4.0, 6.0]),
            "{constructor} dot dimension"
        );
        assert!(
            contains_numeric_tensor(&values, &[1, 2], &[1.0, 7.0]),
            "{constructor} doc2sequence controls"
        );
        assert!(
            values.iter().any(|value| match value {
                Value::CharArray(array) => {
                    array.data.iter().collect::<String>().trim_end() == "01:15:00"
                }
                _ => false,
            }),
            "{constructor} duration matrix"
        );
    }
}

#[test]
fn compiled_double_and_dummyvar_cover_every_integer_class() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for constructor in INTEGER_CONSTRUCTORS {
        let source = format!(
            "converted = double({constructor}([1 3])); encoded = dummyvar({constructor}([1; 2; 1]));"
        );
        let values = execute_source(&source).expect("compiled integer conversion and grouping");
        assert!(
            contains_numeric_tensor(&values, &[1, 2], &[1.0, 3.0]),
            "{constructor} double"
        );
        assert!(
            contains_numeric_tensor(&values, &[3, 2], &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            "{constructor} dummyvar"
        );
    }
}

#[test]
fn compiled_discretize_to_dlmwrite_extensions_reject_in_strict_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "value = dot(uint8([1 2]), uint8([3 4]));",
            "RunMat:compatibility:DotIntegerDataExtension",
        ),
        (
            "value = downsample([1 2 3 4], uint8(2));",
            "RunMat:compatibility:DownsampleIntegerFactorExtension",
        ),
        (
            "value = dummyvar(uint8([1;2]));",
            "RunMat:compatibility:DummyvarIntegerGroupExtension",
        ),
        (
            "value = duration(uint8(1));",
            "RunMat:compatibility:DurationShortComponentFormExtension",
        ),
        (
            "value = double(1, 'like', 0);",
            "RunMat:compatibility:DoubleLikePrototypeExtension",
        ),
        (
            "value = dot(true, true);",
            "RunMat:compatibility:DotLogicalDataExtension",
        ),
        (
            "value = downsample([1 2 3 4], 2, uint8(1));",
            "RunMat:compatibility:DownsampleIntegerPhaseExtension",
        ),
        (
            "value = downsample(reshape([1 2 3 4], [1 2 2]), 2);",
            "RunMat:compatibility:DownsampleNdInputExtension",
        ),
        (
            "value = dlmread('__runmat_integer_missing__.csv', uint8(44));",
            "RunMat:compatibility:DlmreadNumericDelimiterExtension",
        ),
        (
            "value = dlmread('__runmat_integer_missing__.csv', ',', 'A1:B2');",
            "RunMat:compatibility:DlmreadColonSpreadsheetRangeExtension",
        ),
        (
            "value = dlmread('__runmat_integer_missing__.csv', uint8(1), uint8(1));",
            "RunMat:compatibility:DlmreadComposedRangeExtension",
        ),
        (
            "bytes = dlmwrite('/private/tmp/runmat-integer-strict-output.csv', 1);",
            "RunMat:compatibility:DlmwriteByteCountOutputExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict-mode extension rejection");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

struct TempFile(PathBuf);

impl TempFile {
    fn new(extension: &str) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self(std::env::temp_dir().join(format!(
            "runmat-integer-io-{}-{id}.{extension}",
            std::process::id()
        )))
    }

    fn source_literal(&self) -> String {
        self.0
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('\'', "''")
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn compiled_dlmread_accepts_every_integer_offset_class() {
    let input = TempFile::new("csv");
    fs::write(&input.0, "label,a,b\nrow1,1,2\nrow2,3,4\n").expect("write input fixture");
    let filename = input.source_literal();

    for constructor in INTEGER_CONSTRUCTORS {
        let source =
            format!("matrix = dlmread('{filename}', ',', {constructor}(1), {constructor}(1));");
        let values = execute_source(&source).expect("compiled dlmread integer offsets");
        assert!(
            contains_numeric_tensor(&values, &[2, 2], &[1.0, 3.0, 2.0, 4.0]),
            "{constructor} offsets"
        );
    }
}

#[test]
fn compiled_dlmwrite_serializes_every_integer_matrix_class() {
    for constructor in INTEGER_CONSTRUCTORS {
        let output = TempFile::new("csv");
        let filename = output.source_literal();
        let source = format!("dlmwrite('{filename}', {constructor}([1 2; 3 4]));");
        execute_source(&source).expect("compiled dlmwrite integer matrix");
        let contents = fs::read_to_string(&output.0).expect("read dlmwrite output");
        assert_eq!(
            contents.replace("\r\n", "\n"),
            "1,2\n3,4\n",
            "{constructor}"
        );
    }

    let output = TempFile::new("csv");
    let filename = output.source_literal();
    execute_source(&format!(
        "wide = uint64(9007199254740992) + uint64(1); dlmwrite('{filename}', wide);"
    ))
    .expect("compiled exact wide dlmwrite");
    assert_eq!(
        fs::read_to_string(&output.0)
            .expect("read exact wide output")
            .trim(),
        "9007199254740993"
    );
}
