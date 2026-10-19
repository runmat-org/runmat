use super::*;
use crate::context::host_effect::{
    HostEffectAllocationError, HostEffectLiveAllocationClaim, HostEffectLiveAllocationKind,
    HostEffectStageReservation, ManagedHostEffectLedger, ManagedHostEffectStageReservation,
};
use crate::context::RuntimeContext;
use crate::execution::{
    CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError, RuntimeExecutionService,
};
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_filesystem::MemoryFsProvider;
use runmat_value::IntegerStorage;
use runmat_value::{
    ExactHostAllocationPermit, ExactHostAllocationReservation, ExactHostNumericAllocationError,
};
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

#[path = "tests/exact.rs"]
mod exact;

fn fixture_path() -> PathBuf {
    // MemoryFsProvider has a virtual rooted namespace on every platform.
    PathBuf::from("/readmatrix-row/input.csv")
}

fn selected_path(bytes: &[u8]) -> (runmat_filesystem::ProviderGuard, TabularFsPath) {
    let path = fixture_path();
    let provider = Arc::new(MemoryFsProvider::with_current_dir(
        path.parent().expect("fixture parent"),
    ));
    provider.write_project_path(&path, bytes).expect("fixture");
    let restore = runmat_filesystem::replace_provider(provider);
    let selected = TabularFsOwner::capture(Some(Path::new(&path)))
        .expect("captured provider")
        .path(path)
        .expect("selected path");
    (restore, selected)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn offset_range_projects_variable_controls_identically_in_bounded_and_ordinary_readers() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, selected) = selected_path(b"9,N/A,1.234\n8,2,2.500\n");
    let mut missing = VariableImportOptions::new("B".into(), "double".into());
    missing
        .apply("TreatAsMissing", &Value::from("N/A"))
        .unwrap();
    missing.apply("FillValue", &Value::Num(0.0)).unwrap();
    let mut grouped = VariableImportOptions::new("C".into(), "double".into());
    grouped
        .apply("DecimalSeparator", &Value::from(","))
        .unwrap();
    grouped
        .apply("ThousandsSeparator", &Value::from("."))
        .unwrap();
    let options = ReadMatrixOptions {
        delimiter: Some(Delimiter::Char(',')),
        range: Some(RangeSpec {
            start_row: 0,
            start_col: 1,
            end_row: None,
            end_col: Some(2),
        }),
        variable_options: vec![
            VariableImportOptions::new("A".into(), "double".into()),
            missing,
            grouped,
        ],
        ..ReadMatrixOptions::default()
    };
    options.validate().expect("admitted column projection");
    assert_eq!(options.column_from_source(0).selected_index(), None);
    assert_eq!(options.column_from_source(1).selected_index(), Some(0));
    assert_eq!(options.column_from_selected(1).source_index(), 2);

    let mut bounded_source = block_on(selected.open_bounded_input())
        .expect("bounded open")
        .expect("memory snapshot");
    let mut bounded_work = TabularCooperation::new(None, BUILTIN_NAME);
    let geometry = block_on(bounded::scan_utf8(
        &mut bounded_source,
        &mut bounded_work,
        &selected,
        &options,
        bounded::ScanResources::default(),
    ))
    .expect("bounded scan");
    let bounded_output = block_on(bounded::fill_utf8(
        &mut bounded_source,
        &geometry,
        &mut bounded_work,
        &selected,
        &options,
        bounded::FillResources::default(),
    ))
    .expect("bounded selected output");

    let ordinary_file = block_on(selected.open_read()).expect("ordinary open");
    let mut ordinary_work = TabularCooperation::new(None, BUILTIN_NAME);
    let ordinary_output = block_on(read_numeric_matrix_from_bytes(
        &mut ordinary_work,
        &selected,
        &options,
        TabularReadSource::Ordinary(ordinary_file),
    ))
    .expect("ordinary selected output");
    assert_eq!(bounded_output.shape(), &[2, 2]);
    assert_eq!(
        bounded_output.materialize_f64(),
        vec![0.0, 2.0, 1234.0, 2500.0]
    );
    assert_eq!(ordinary_output, bounded_output);
    let mut range_detected = options.clone();
    range_detected.variable_source_origin = Some(1);
    range_detected.variable_options.remove(0);
    assert_eq!(
        range_detected
            .variable(range_detected.column_from_selected(0))
            .unwrap()
            .name,
        "B"
    );
    assert_eq!(
        range_detected
            .variable(range_detected.column_from_selected(1))
            .unwrap()
            .name,
        "C"
    );
    let mut selected_legacy = range_detected;
    selected_legacy.variable_source_origin = None;
    assert_eq!(
        selected_legacy
            .variable(selected_legacy.column_from_selected(0))
            .unwrap()
            .name,
        "B"
    );
    assert_eq!(
        selected_legacy
            .variable(selected_legacy.column_from_selected(1))
            .unwrap()
            .name,
        "C"
    );
    let mut explicit_zero = selected_legacy;
    explicit_zero.variable_source_origin = Some(0);
    assert!(explicit_zero
        .variable(explicit_zero.column_from_selected(1))
        .is_none());
}

struct TestDiagnosticStage(Arc<ManagedHostEffectStageReservation>);

impl DiagnosticReservation for TestDiagnosticStage {
    fn reserve_diagnostic_claim(
        &self,
        bytes: usize,
    ) -> Result<HostEffectLiveAllocationClaim, HostEffectAllocationError> {
        self.0
            .reserve_live_exact_claim(HostEffectLiveAllocationKind::Diagnostic, bytes)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn staged_scan_retains_full_first_numeric_error_through_remaining_utf8_validation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let bytes = b"1,abc\n2,3\n";
    let (_restore, selected) = selected_path(bytes);
    let mut source = block_on(selected.open_bounded_input())
        .expect("bounded source")
        .expect("memory snapshot");
    let ledger = ManagedHostEffectLedger::new(4096);
    let stage = ledger.reserve_stage(4096).expect("diagnostic stage");
    let diagnostic = TestDiagnosticStage(stage.clone());
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let Err(error) = block_on(bounded::scan_utf8(
        &mut source,
        &mut cooperation,
        &selected,
        &ReadMatrixOptions::default(),
        bounded::ScanResources {
            line: Some(crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len())),
            samples: Some(
                crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len()),
            ),
            field: Some(
                crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len()),
            ),
            diagnostic: Some(&diagnostic),
        },
    )) else {
        panic!("expected numeric error after complete UTF-8 scan");
    };
    assert_eq!(
        error.message(),
        "readmatrix: unable to parse numeric value 'abc' on line 1 column 2"
    );
    assert_eq!(
        error.identifier(),
        READMATRIX_ERROR_NON_NUMERIC_TOKEN.identifier
    );
    assert!(error.retained_diagnostic_allocation.is_some());
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(diagnostic);
    drop(stage);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(error);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn staged_scan_incomplete_utf8_owns_fixed_error_until_last_drop() {
    let _lock = runmat_filesystem::provider_override_lock();
    let bytes = b"1,\xc3";
    let (_restore, selected) = selected_path(bytes);
    let mut source = block_on(selected.open_bounded_input())
        .expect("bounded source")
        .expect("memory snapshot");
    let ledger = ManagedHostEffectLedger::new(4096);
    let stage = ledger.reserve_stage(4096).expect("diagnostic stage");
    let diagnostic = TestDiagnosticStage(stage.clone());
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let Err(error) = block_on(bounded::scan_utf8(
        &mut source,
        &mut cooperation,
        &selected,
        &ReadMatrixOptions::default(),
        bounded::ScanResources {
            line: Some(crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len())),
            samples: Some(
                crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len()),
            ),
            field: Some(
                crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len()),
            ),
            diagnostic: Some(&diagnostic),
        },
    )) else {
        panic!("incomplete UTF-8 must fail before numeric parsing");
    };
    assert_eq!(
        error.message(),
        "readmatrix: file contains bytes that are invalid for Encoding 'utf-8'"
    );
    assert_eq!(error.identifier(), READMATRIX_ERROR_IO_READ.identifier);
    assert!(error.retained_diagnostic_allocation.is_some());
    drop(diagnostic);
    drop(stage);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(error);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn staged_scan_invalid_utf8_retains_exact_typed_source_and_charge() {
    let _lock = runmat_filesystem::provider_override_lock();
    let bytes = b"1,\xff";
    let (_restore, selected) = selected_path(bytes);
    let mut source = block_on(selected.open_bounded_input())
        .expect("bounded source")
        .expect("memory snapshot");
    let ledger = ManagedHostEffectLedger::new(4096);
    let stage = ledger.reserve_stage(4096).expect("diagnostic stage");
    let diagnostic = TestDiagnosticStage(stage.clone());
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let Err(error) = block_on(bounded::scan_utf8(
        &mut source,
        &mut cooperation,
        &selected,
        &ReadMatrixOptions::default(),
        bounded::ScanResources {
            line: Some(crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len())),
            samples: Some(
                crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len()),
            ),
            field: Some(
                crate::context::host_effect::ExactHostEffectBytes::test_buffer(bytes.len()),
            ),
            diagnostic: Some(&diagnostic),
        },
    )) else {
        panic!("invalid UTF-8 has an exact typed parse cause");
    };
    assert_eq!(
        error.message(),
        "readmatrix: file contains bytes that are invalid for Encoding 'utf-8'"
    );
    assert_eq!(error.identifier(), READMATRIX_ERROR_IO_READ.identifier);
    assert!(error
        .source
        .as_deref()
        .is_some_and(|source| source.is::<std::str::Utf8Error>()));
    assert!(error.retained_diagnostic_allocation.is_some());
    let payload = error.message.capacity()
        + error.identifier.as_ref().map_or(0, String::capacity)
        + error.context.builtin.as_ref().unwrap().capacity()
        + std::mem::size_of::<std::str::Utf8Error>();
    assert_eq!(
        ledger.snapshot().live_claimed_bytes,
        (payload + stage.live_claim_metadata_bytes()) as u64
    );
    drop(diagnostic);
    drop(stage);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(error);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
}

#[derive(Debug)]
struct ExactOutputTestPermit(Arc<AtomicUsize>);

#[derive(Debug)]
struct ExactOutputTestReservation {
    live: Arc<AtomicUsize>,
    bytes: usize,
}

impl Drop for ExactOutputTestReservation {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

impl ExactHostAllocationReservation for ExactOutputTestReservation {}

impl ExactHostAllocationPermit for ExactOutputTestPermit {
    fn reserve_exact(
        &self,
        bytes: usize,
    ) -> Result<Arc<dyn ExactHostAllocationReservation>, ExactHostNumericAllocationError> {
        self.0.fetch_add(bytes, Ordering::SeqCst);
        Ok(Arc::new(ExactOutputTestReservation {
            live: self.0.clone(),
            bytes,
        }))
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn retained_utf8_replay_writes_exact_wide_integer_output_without_mirror() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, selected) = selected_path(b"9007199254740993,18446744073709551615\n");
    let mut source = block_on(selected.open_bounded_input())
        .expect("bounded opened source")
        .expect("direct memory snapshot");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let options = ReadMatrixOptions {
        delimiter: Some(Delimiter::Char(',')),
        output_template: OutputTemplate::Numeric(NumericDType::U64),
        ..ReadMatrixOptions::default()
    };
    let geometry = block_on(bounded::scan_utf8(
        &mut source,
        &mut cooperation,
        &selected,
        &options,
        bounded::ScanResources::default(),
    ))
    .expect("first pass");
    let live = Arc::new(AtomicUsize::new(0));
    let permit = ExactOutputTestPermit(live.clone());
    let tensor = block_on(bounded::fill_utf8(
        &mut source,
        &geometry,
        &mut cooperation,
        &selected,
        &options,
        bounded::FillResources {
            output: Some(&permit),
            ..Default::default()
        },
    ))
    .expect("exact replay");
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(
        tensor.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );
    let exact_bytes =
        runmat_value::ExactHostTensor2dAllocation::requested_bytes(NumericDType::U64, 1, 2)
            .expect("canonical exact tensor owner layout");
    assert_eq!(live.load(Ordering::SeqCst), exact_bytes);
    drop(tensor);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn borrowed_field_scanner_preserves_quoted_and_multichar_forms() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, path) = selected_path(b"1\n");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let quoted = block_on(scan::split_fields(
        "\"1,2\",3,\"4\"\"5\"",
        &Delimiter::Char(','),
        &mut cooperation,
        &path,
    ))
    .expect("quoted cells");
    assert_eq!(
        quoted
            .iter()
            .map(|field| field.as_ref())
            .collect::<Vec<_>>(),
        vec!["1,2", "3", "4\"5"]
    );
    let multi = block_on(scan::split_fields(
        "a::b::::c",
        &Delimiter::String("::".into()),
        &mut cooperation,
        &path,
    ))
    .expect("multi-character delimiter");
    assert_eq!(
        multi.iter().map(|field| field.as_ref()).collect::<Vec<_>>(),
        vec!["a", "b", "", "c"]
    );
    let whitespace = block_on(scan::split_fields(
        "  1\u{2003}2\t3  ",
        &Delimiter::Whitespace,
        &mut cooperation,
        &path,
    ))
    .expect("Unicode whitespace");
    assert_eq!(
        whitespace
            .iter()
            .map(|field| field.as_ref())
            .collect::<Vec<_>>(),
        vec!["1", "2", "3"]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn staged_field_scanner_reuses_exact_scratch_without_retaining_field_text() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, path) = selected_path(b"1\n");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let mut scratch = crate::context::host_effect::ExactHostEffectBytes::test_buffer(64);
    scratch.append(&[0xff]).expect("seed invalid scratch bytes");
    let mut seen = Vec::new();
    let count = block_on(scan::for_each_field_ref(
        "\"1,2\",3,\"4\"\"5\"",
        &Delimiter::Char(','),
        &mut cooperation,
        &path,
        Some(&mut scratch),
        |_, field| {
            seen.push(field.as_str().to_owned());
            Ok(())
        },
    ))
    .expect("staged quoted fields");
    assert_eq!(count, 3);
    assert_eq!(seen, ["1,2", "3", "4\"5"]);
    assert!(scratch.as_bytes().is_empty());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn numeric_token_borrowed_fast_path_preserves_case_and_separator_semantics() {
    let default = ReadMatrixOptions::default();
    assert!(
        parse_numeric_token("nAn", &default, 1, default.column_from_source(0))
            .expect("mixed-case NaN")
            .is_nan()
    );
    assert_eq!(
        parse_numeric_token("+InFiNiTy", &default, 1, default.column_from_source(1))
            .expect("mixed-case infinity"),
        f64::INFINITY
    );
    assert_eq!(
        parse_integer_numeric_token(
            "9007199254740993",
            &default,
            NumericDType::U64,
            1,
            default.column_from_source(2)
        )
        .expect("exact wide integer"),
        NumericScalar::U64(9_007_199_254_740_993)
    );
    let nondefault = ReadMatrixOptions {
        decimal_separator: ',',
        thousands_separator: Some('.'),
        ..ReadMatrixOptions::default()
    };
    assert_eq!(
        parse_numeric_token("1.234,5", &nondefault, 2, nondefault.column_from_source(0))
            .expect("grouping and decimal"),
        1_234.5
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn borrowed_rows_retain_exact_wide_integer_range_and_row_diagnostics() {
    let _lock = runmat_filesystem::provider_override_lock();
    let (_restore, path) = selected_path(b"0,0\n9007199254740993,18446744073709551615\n");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let mut options = ReadMatrixOptions::default();
    options
        .set_output_type("uint64")
        .expect("typed output control");
    options.range = Some(super::super::range::parse_range(&Value::from("A2:B2")).expect("range"));
    let matrix = block_on(read_numeric_matrix(None, &mut cooperation, &path, &options))
        .expect("selected exact integers");
    assert_eq!(matrix.shape(), &[1, 2]);
    assert_eq!(
        matrix.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );

    let (_restore, bad_path) = selected_path(b"1,2\n3,not-a-number\n");
    let error = block_on(read_numeric_matrix(
        None,
        &mut cooperation,
        &bad_path,
        &ReadMatrixOptions::default(),
    ))
    .expect_err("invalid second-row field");
    assert!(error.message().contains("line 2 column 2"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn retained_seekable_utf8_scan_auto_detects_delimiter_without_reopening() {
    let _lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("isolated numeric-text source");
    let file = directory.path().join("numbers.txt");
    std::fs::write(&file, b"1;2\r\n3;4\r\n").expect("UTF-8 fixture");
    let _restore =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
    let selected = TabularFsOwner::capture(Some(&file))
        .expect("captured provider")
        .path(file)
        .expect("selected native source");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let matrix = block_on(read_numeric_matrix(
        None,
        &mut cooperation,
        &selected,
        &ReadMatrixOptions::default(),
    ))
    .expect("detected semicolon matrix");
    assert_eq!(matrix.shape(), &[2, 2]);
    assert_eq!(matrix.as_f64_slice(), Some(&[1.0, 3.0, 2.0, 4.0][..]));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn retained_utf8_row_trims_long_cr_and_bom_runs_with_cooperation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("isolated numeric-text source");
    let file = directory.path().join("long-row.csv");
    let mut input = "\u{FEFF}".repeat(4_096).into_bytes();
    input.extend_from_slice(b"1,2");
    input.extend(vec![b'\r'; 16_384]);
    input.push(b'\n');
    std::fs::write(&file, input).expect("long UTF-8 fixture");
    let _restore =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
    let selected = TabularFsOwner::capture(Some(&file))
        .expect("captured provider")
        .path(file)
        .expect("selected native source");
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let turns = Rc::new(Cell::new(0usize));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4_096).expect("nonzero")),
        runtime.cancellation(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            counted.set(counted.get() + 1);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let options = ReadMatrixOptions {
        delimiter: Some(Delimiter::Char(',')),
        ..ReadMatrixOptions::default()
    };
    let tensor = block_on(read_numeric_matrix(
        Some(&runtime),
        &mut cooperation,
        &selected,
        &options,
    ))
    .expect("long CR/BOM row");
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.as_f64_slice(), Some(&[1.0, 2.0][..]));
    assert!(turns.get() >= 4, "long row reached execution-service turns");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn retained_seekable_utf8_scan_decodes_multibyte_header_across_chunk() {
    let _lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("isolated numeric-text source");
    let file = directory.path().join("numbers.txt");
    let mut input = vec![b'x'; 4094];
    input.extend_from_slice("😀\n1,2\n".as_bytes());
    std::fs::write(&file, input).expect("UTF-8 fixture");
    let _restore =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
    let selected = TabularFsOwner::capture(Some(&file))
        .expect("captured provider")
        .path(file)
        .expect("selected native source");
    let options = ReadMatrixOptions {
        num_header_lines: 1,
        delimiter: Some(Delimiter::Char(',')),
        ..ReadMatrixOptions::default()
    };
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let matrix = block_on(read_numeric_matrix(
        None,
        &mut cooperation,
        &selected,
        &options,
    ))
    .expect("valid split UTF-8 code point");
    assert_eq!(matrix.shape(), &[1, 2]);
    assert_eq!(matrix.as_f64_slice(), Some(&[1.0, 2.0][..]));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn retained_seekable_utf8_scan_rejects_same_shape_in_place_mutation() {
    let _lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("isolated numeric-text source");
    let file = directory.path().join("numbers.csv");
    std::fs::write(&file, b"1,2\n3,4\n").expect("initial payload");
    let _restore =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
    let selected = TabularFsOwner::capture(Some(&file))
        .expect("captured provider")
        .path(file.clone())
        .expect("selected source");
    let mut opened = block_on(selected.open_bounded_input())
        .expect("native direct read")
        .expect("bounded source");
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let options = ReadMatrixOptions {
        delimiter: Some(Delimiter::Char(',')),
        ..ReadMatrixOptions::default()
    };
    let geometry = block_on(bounded::scan_utf8(
        &mut opened,
        &mut cooperation,
        &selected,
        &options,
        bounded::ScanResources::default(),
    ))
    .expect("first pass");
    std::fs::write(&file, b"9,8\n7,6\n").expect("same-length in-place mutation");
    let error = block_on(bounded::fill_utf8(
        &mut opened,
        &geometry,
        &mut cooperation,
        &selected,
        &options,
        bounded::FillResources::default(),
    ))
    .expect_err("second pass cannot publish unvalidated new bytes");
    assert!(error
        .message()
        .contains("changed between validation and output"));

    std::fs::write(&file, b"1,2\n3,4\n").expect("reset payload");
    let mut opened = block_on(selected.open_bounded_input())
        .expect("direct read")
        .expect("bounded source");
    let geometry = block_on(bounded::scan_utf8(
        &mut opened,
        &mut cooperation,
        &selected,
        &options,
        bounded::ScanResources::default(),
    ))
    .expect("new first pass");
    std::fs::write(&file, b"x,2\n3,4\n").expect("same-length malformed mutation");
    let ledger = ManagedHostEffectLedger::new(4096);
    let stage = ledger.reserve_stage(4096).expect("diagnostic stage");
    let diagnostic = TestDiagnosticStage(stage.clone());
    let error = block_on(bounded::fill_utf8(
        &mut opened,
        &geometry,
        &mut cooperation,
        &selected,
        &options,
        bounded::FillResources {
            diagnostic: Some(&diagnostic),
            ..Default::default()
        },
    ))
    .expect_err("changed-source error precedes replay token error");
    assert!(error
        .message()
        .contains("changed between validation and output"));
    assert!(error.retained_diagnostic_allocation.is_some());
    assert!(ledger.snapshot().live_claimed_bytes > 0);
    drop(diagnostic);
    drop(stage);
    assert!(ledger.snapshot().live_reserved_bytes > 0);
    drop(error);
    assert_eq!(ledger.snapshot().live_claimed_bytes, 0);
    assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

    std::fs::write(&file, b"1,2\n3,4\n").expect("reset payload");
    let mut opened = block_on(selected.open_bounded_input())
        .expect("direct read")
        .expect("bounded source");
    let mut empty_options = options.clone();
    empty_options.range = Some(super::super::range::parse_range(&Value::from("Z9")).unwrap());
    let geometry = block_on(bounded::scan_utf8(
        &mut opened,
        &mut cooperation,
        &selected,
        &empty_options,
        bounded::ScanResources::default(),
    ))
    .expect("out-of-range first pass");
    std::fs::write(&file, b"9,8\n7,6\n").expect("same-length mutation outside range");
    let error = block_on(bounded::fill_utf8(
        &mut opened,
        &geometry,
        &mut cooperation,
        &selected,
        &empty_options,
        bounded::FillResources::default(),
    ))
    .expect_err("empty range still checks retained source before publication");
    assert!(error
        .message()
        .contains("changed between validation and output"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn retained_seekable_utf8_scan_preserves_bom_range_and_exact_wide_integer() {
    let _lock = runmat_filesystem::provider_override_lock();
    let directory = tempfile::tempdir().expect("isolated numeric-text source");
    let file = directory.path().join("numbers.csv");
    std::fs::write(
        &file,
        b"\xef\xbb\xbfskip,skip\r\n1,2\r\n9007199254740993,18446744073709551615\r\n",
    )
    .expect("UTF-8 fixture");
    let _restore =
        runmat_filesystem::replace_provider(Arc::new(runmat_filesystem::NativeFsProvider));
    let selected = TabularFsOwner::capture(Some(&file))
        .expect("captured provider")
        .path(file.clone())
        .expect("selected native source");
    let mut options = ReadMatrixOptions {
        encoding: "utf-8".into(),
        num_header_lines: 1,
        delimiter: Some(Delimiter::Char(',')),
        ..ReadMatrixOptions::default()
    };
    options.set_output_type("uint64").expect("typed output");
    options.range = Some(super::super::range::parse_range(&Value::from("A2:B2")).unwrap());
    let mut cooperation = TabularCooperation::new(None, BUILTIN_NAME);
    let matrix = block_on(read_numeric_matrix(
        None,
        &mut cooperation,
        &selected,
        &options,
    ))
    .expect("selected exact integer row");
    assert_eq!(matrix.shape(), &[1, 2]);
    assert_eq!(
        matrix.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]))
    );

    std::fs::write(&file, b"1,2\n3,not-a-number\n").expect("replacement fixture");
    let error = block_on(read_numeric_matrix(
        None,
        &mut cooperation,
        &selected,
        &options,
    ))
    .expect_err("first pass validates even rows outside requested Range");
    assert!(error.message().contains("line 2 column 2"));

    let mut malformed = b"skip,skip\nnot-a-number,1\n".to_vec();
    for _ in 0..31 {
        malformed.extend_from_slice(b"1,2\n");
    }
    malformed.extend_from_slice(b"1,\xff\n");
    std::fs::write(&file, malformed).expect("later invalid encoding fixture");
    let error = block_on(read_numeric_matrix(
        None,
        &mut cooperation,
        &selected,
        &options,
    ))
    .expect_err("whole-source encoding error precedes numeric token error");
    assert!(error.message().contains("invalid for Encoding 'utf-8'"));
}

struct PendingTurn(Rc<Cell<bool>>);

impl Future for PendingTurn {
    type Output = Result<(), ExecutionServiceError>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingTurn {
    fn drop(&mut self) {
        self.0.set(true);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelling_pending_single_wide_row_scan_releases_work_without_output() {
    let _lock = runmat_filesystem::provider_override_lock();
    let mut input = vec![b'1'; 32_767];
    for index in (1..input.len()).step_by(2) {
        input[index] = b',';
    }
    input.push(b'\n');
    let (_restore, path) = selected_path(&input);
    let source = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(RuntimeExecutionService::new()), source.token());
    let turns = Rc::new(Cell::new(0usize));
    let dropped = Rc::new(Cell::new(false));
    let turn_count = turns.clone();
    let turn_dropped = dropped.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(4_096).expect("nonzero")),
        source.token(),
        move || -> crate::context::RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
            let count = turn_count.get() + 1;
            turn_count.set(count);
            if count < 11 {
                Box::pin(async { Ok(()) })
            } else {
                Box::pin(PendingTurn(turn_dropped.clone()))
            }
        },
    );
    let mut cooperation = TabularCooperation::with_test_budget(&runtime, BUILTIN_NAME, budget);
    let options = ReadMatrixOptions {
        delimiter: Some(Delimiter::Char(',')),
        ..ReadMatrixOptions::default()
    };
    let mut call = Box::pin(read_numeric_matrix(
        Some(&runtime),
        &mut cooperation,
        &path,
        &options,
    ));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert_eq!(
        turns.get(),
        11,
        "one wide row reached incremental parsing work"
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = block_on(call).expect_err("cancelled row cannot publish a tensor");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert!(dropped.get(), "pending service turn was released");
}
