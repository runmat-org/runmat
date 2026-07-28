#[path = "support/mod.rs"]
mod test_helpers;

use std::sync::atomic::{AtomicU64, Ordering};

use runmat_time::unix_timestamp_ms;
use test_helpers::execute_source;

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn temp_xlsx_path() -> std::path::PathBuf {
    let millis = unix_timestamp_ms();
    let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_vm_xlswrite_{}_{}_{}.xlsx",
        std::process::id(),
        millis,
        unique
    ));
    path
}

#[test]
fn xlswrite_round_trips_numeric_matrix_through_xlsread() {
    let path = temp_xlsx_path();
    let filename = path.to_string_lossy().replace('\'', "''");
    let source = format!(
        "\
        [ok, msg] = xlswrite('{filename}', [1 2 3; 4 5 6], 'Data', 'B2'); \
        if ~ok; error(msg.message); end; \
        num = xlsread('{filename}', 'Data', 'B2:D3'); \
        if size(num, 1) ~= 2 || size(num, 2) ~= 3; \
            error('xlswrite round-trip shape mismatch'); \
        end; \
        expected = [1 2 3; 4 5 6]; \
        if max(abs(num(:) - expected(:))) > 1e-12; \
            error('xlswrite round-trip values mismatch'); \
        end;"
    );

    execute_source(&source).expect("execute xlswrite round trip");
    let _ = std::fs::remove_file(path);
}

#[test]
fn xlswrite_numeric_sheet_ordinal_round_trips_through_xlsread() {
    let path = temp_xlsx_path();
    let filename = path.to_string_lossy().replace('\'', "''");
    let source = format!(
        "\
        ok = xlswrite('{filename}', 11, 'First', 'A1'); \
        if ~ok; error('first xlswrite failed'); end; \
        ok = xlswrite('{filename}', 22, 2, 'B2'); \
        if ~ok; error('second xlswrite failed'); end; \
        first = xlsread('{filename}', 1, 'A1'); \
        second = xlsread('{filename}', 2, 'B2'); \
        if first ~= 11 || second ~= 22; \
            error('xlswrite numeric sheet ordinal mismatch'); \
        end;"
    );

    execute_source(&source).expect("execute xlswrite numeric sheet round trip");
    let _ = std::fs::remove_file(path);
}
