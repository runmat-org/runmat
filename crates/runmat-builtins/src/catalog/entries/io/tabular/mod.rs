//! Typed catalog authority for tabular file I/O.

mod csvread;
mod parquetread;
mod parquetinfo;
mod parquet_fixture;
mod csvwrite;
mod dlmread;
mod dlmwrite;
mod readmatrix;
mod import_options;
mod writematrix;
mod writecell;
mod xlswrite;
mod xlsread;
mod reader_contract;
mod readtable;
mod readcell;
mod spreadsheet_import_options;
mod detect_import_options;

pub use csvread::*;
pub use csvwrite::*;
pub use dlmread::*;
pub use dlmwrite::*;
pub use import_options::*;
pub use parquetinfo::*;
pub use parquetread::*;
pub use readmatrix::*;
pub use writecell::*;
pub use writematrix::*;
pub use xlsread::*;
pub use xlswrite::*;
pub use reader_contract::*;
pub use readtable::*;
pub use readcell::*;
pub use spreadsheet_import_options::*;
pub use detect_import_options::*;

pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    values.extend(csvread::ENTRIES.iter().copied());
    values.extend(parquetread::ENTRIES.iter().copied());
    values.extend(parquetinfo::ENTRIES.iter().copied());
    values.extend(csvwrite::ENTRIES.iter().copied());
    values.extend(dlmread::ENTRIES.iter().copied());
    values.extend(dlmwrite::ENTRIES.iter().copied());
    values.extend(readmatrix::ENTRIES.iter().copied());
    import_options::extend_entries(values);
    values.extend(writematrix::ENTRIES.iter().copied());
    values.extend(writecell::ENTRIES.iter().copied());
    values.extend(xlswrite::ENTRIES.iter().copied());
    values.extend(xlsread::ENTRIES.iter().copied());
    values.extend(readtable::ENTRIES.iter().copied());
    values.extend(readcell::ENTRIES.iter().copied());
    values.extend(spreadsheet_import_options::ENTRIES.iter().copied());
    values.extend(detect_import_options::ENTRIES.iter().copied());
}
