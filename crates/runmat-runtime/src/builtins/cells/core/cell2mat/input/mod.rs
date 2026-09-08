mod parse;
mod prototype;

use runmat_value::{IntegerComplexStorage, NumericStorage};

use super::error::{cell2mat_error_with_message, CELL2MAT_ERROR_INVALID_CONTENTS};
use crate::{gather_if_needed_async, BuiltinResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ElementKind {
    Numeric,
    Complex,
    TypedComplexInteger,
    Logical,
    Character,
}

#[derive(Clone)]
pub(super) struct CellEntry {
    pub(super) kind: ElementKind,
    pub(super) shape: Vec<usize>,
    pub(super) data: EntryData,
}

#[derive(Clone)]
pub(super) enum EntryData {
    Numeric(NumericStorage),
    Complex(Vec<(f64, f64)>),
    TypedComplexInteger(IntegerComplexStorage),
    Logical(Vec<u8>),
    Character(Vec<char>),
}

impl EntryData {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Numeric(storage) => storage.len(),
            Self::Complex(data) => data.len(),
            Self::TypedComplexInteger(storage) => storage.len(),
            Self::Logical(data) => data.len(),
            Self::Character(data) => data.len(),
        }
    }
}

impl CellEntry {
    pub(super) fn len(&self) -> usize {
        self.data.len()
    }
}

pub(super) async fn gather(
    cells: &runmat_value::CellArray,
) -> BuiltinResult<(ElementKind, Vec<CellEntry>)> {
    let mut entries = Vec::with_capacity(cells.data.len());
    let mut kind = None;
    for value in &cells.data {
        let entry = parse::value(gather_if_needed_async(value).await?)?;
        if entry.len() > 0 {
            if kind.is_some_and(|expected| expected != entry.kind) {
                return Err(cell2mat_error_with_message(
                    "cell2mat: all non-empty cell contents must share a fundamental type",
                    &CELL2MAT_ERROR_INVALID_CONTENTS,
                ));
            }
            kind.get_or_insert(entry.kind);
        }
        entries.push(entry);
    }
    Ok((kind.unwrap_or(ElementKind::Numeric), entries))
}

pub(super) use prototype::{numeric as numeric_prototype, typed_complex_integer};
