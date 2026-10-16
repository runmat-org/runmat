//! One source-to-selected column projection for both reader implementations.
//! Diagnostics and import-option variables retain the physical file column;
//! the selected index belongs only to output geometry.

use super::ReadMatrixOptions;

#[derive(Clone, Copy)]
pub(super) struct MatrixFieldColumn {
    source_zero_based: usize,
    selected_zero_based: Option<usize>,
}

impl MatrixFieldColumn {
    pub(super) fn source_index(self) -> usize {
        self.source_zero_based
    }

    pub(super) fn diagnostic_number(self) -> usize {
        self.source_zero_based + 1
    }

    pub(super) fn selected_index(self) -> Option<usize> {
        self.selected_zero_based
    }
}

impl ReadMatrixOptions {
    pub(super) fn column_from_source(&self, source_zero_based: usize) -> MatrixFieldColumn {
        let selected_zero_based = match &self.range {
            Some(range) => source_zero_based
                .checked_sub(range.start_col)
                .filter(|_| range.end_col.is_none_or(|end| source_zero_based <= end)),
            None => Some(source_zero_based),
        };
        MatrixFieldColumn {
            source_zero_based,
            selected_zero_based,
        }
    }

    pub(super) fn column_from_selected(&self, selected_zero_based: usize) -> MatrixFieldColumn {
        let source_zero_based = self
            .range
            .as_ref()
            .map_or(0, |range| range.start_col)
            .checked_add(selected_zero_based)
            .expect("selected output column must lie in validated source geometry");
        MatrixFieldColumn {
            source_zero_based,
            selected_zero_based: Some(selected_zero_based),
        }
    }
}
