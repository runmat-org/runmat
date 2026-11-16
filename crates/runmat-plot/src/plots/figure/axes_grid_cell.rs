use serde::{Deserialize, Serialize};

/// A rectangular subplot region in its own row-major grid.
///
/// `index` names the top-left slot. Spans include that slot and stay within
/// the grid. Single-cell records from older scenes omit the two span fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AxesGridCell {
    pub rows: u32,
    pub columns: u32,
    pub index: u32,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub row_span: u32,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub column_span: u32,
}

const fn one() -> u32 {
    1
}

fn is_one(value: &u32) -> bool {
    *value == 1
}

impl AxesGridCell {
    pub fn single(rows: u32, columns: u32, index: u32) -> Self {
        Self {
            rows,
            columns,
            index,
            row_span: 1,
            column_span: 1,
        }
    }

    /// Cover the bounding rectangle of zero-based slots, without allocating
    /// a grid or retaining the input list.
    pub fn from_slots(
        rows: u32,
        columns: u32,
        slots: impl IntoIterator<Item = u32>,
    ) -> Option<Self> {
        let count = rows.checked_mul(columns)?;
        if count == 0 {
            return None;
        }
        let mut slots = slots.into_iter();
        let first = slots.next()?;
        if first >= count {
            return None;
        }
        let (mut min_row, mut max_row) = (first / columns, first / columns);
        let (mut min_column, mut max_column) = (first % columns, first % columns);
        for slot in slots {
            if slot >= count {
                return None;
            }
            let (row, column) = (slot / columns, slot % columns);
            min_row = min_row.min(row);
            max_row = max_row.max(row);
            min_column = min_column.min(column);
            max_column = max_column.max(column);
        }
        Some(Self {
            rows,
            columns,
            index: min_row * columns + min_column,
            row_span: max_row - min_row + 1,
            column_span: max_column - min_column + 1,
        })
    }

    pub fn is_valid(self) -> bool {
        let Some(count) = self.rows.checked_mul(self.columns) else {
            return false;
        };
        count > 0
            && self.index < count
            && self.row_span > 0
            && self.column_span > 0
            && self.row_span <= self.rows - self.index / self.columns
            && self.column_span <= self.columns - self.index % self.columns
    }

    pub fn including_slot(self, slot: u32) -> Option<Self> {
        if !self.is_valid() {
            return None;
        }
        let last = self.index + (self.row_span - 1) * self.columns + self.column_span - 1;
        Self::from_slots(self.rows, self.columns, [self.index, last, slot])
    }

    /// Normalized left, bottom, width and height in figure coordinates.
    pub fn normalized_position(self) -> Option<[f64; 4]> {
        if !self.is_valid() {
            return None;
        }
        let row = self.index / self.columns;
        let column = self.index % self.columns;
        Some([
            f64::from(column) / f64::from(self.columns),
            f64::from(self.rows - row - self.row_span) / f64::from(self.rows),
            f64::from(self.column_span) / f64::from(self.columns),
            f64::from(self.row_span) / f64::from(self.rows),
        ])
    }

    /// Whether two grid descriptions select the same physical rectangle.
    pub fn same_region(self, other: Self) -> bool {
        if !(self.is_valid() && other.is_valid()) {
            return false;
        }
        let row = self.index / self.columns;
        let column = self.index % self.columns;
        let other_row = other.index / other.columns;
        let other_column = other.index % other.columns;
        let equal = |a: u32, a_denominator: u32, b: u32, b_denominator: u32| {
            u64::from(a) * u64::from(b_denominator) == u64::from(b) * u64::from(a_denominator)
        };
        equal(row, self.rows, other_row, other.rows)
            && equal(self.row_span, self.rows, other.row_span, other.rows)
            && equal(column, self.columns, other_column, other.columns)
            && equal(
                self.column_span,
                self.columns,
                other.column_span,
                other.columns,
            )
    }

    /// Compare exact grid fractions, including layouts expressed on different
    /// grids. Edge contact alone does not count as overlap.
    pub fn overlaps(self, other: Self) -> bool {
        if !(self.is_valid() && other.is_valid()) {
            return false;
        }
        let row = self.index / self.columns;
        let column = self.index % self.columns;
        let other_row = other.index / other.columns;
        let other_column = other.index % other.columns;
        let less = |a: u32, a_denominator: u32, b: u32, b_denominator: u32| {
            u64::from(a) * u64::from(b_denominator) < u64::from(b) * u64::from(a_denominator)
        };
        less(row, self.rows, other_row + other.row_span, other.rows)
            && less(other_row, other.rows, row + self.row_span, self.rows)
            && less(
                column,
                self.columns,
                other_column + other.column_span,
                other.columns,
            )
            && less(
                other_column,
                other.columns,
                column + self.column_span,
                self.columns,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_cover_slot_bounds_and_keep_mixed_grid_neighbors_separate() {
        let left = AxesGridCell::from_slots(2, 3, [0, 1, 3, 4]).unwrap();
        assert_eq!(left.normalized_position(), Some([0.0, 0.0, 2.0 / 3.0, 1.0]));
        assert!(!left.overlaps(AxesGridCell::single(2, 3, 2)));
        assert!(!left.overlaps(AxesGridCell::single(2, 3, 5)));
        assert_eq!(AxesGridCell::from_slots(2, 3, [4, 0]), Some(left));
        assert_eq!(AxesGridCell::single(2, 3, 0).including_slot(4), Some(left));
        assert_eq!(left.including_slot(6), None);
        let half = AxesGridCell::single(1, 2, 0);
        assert!(!half.overlaps(AxesGridCell::single(2, 2, 1)));
        assert!(!half.overlaps(AxesGridCell::single(2, 2, 3)));
        assert!(half.overlaps(AxesGridCell::single(3, 3, 1)));
        assert!(half.same_region(AxesGridCell::from_slots(2, 4, [0, 1, 4, 5]).unwrap()));
        assert!(!half.same_region(left));
    }

    #[test]
    fn scene_cells_default_to_one_slot_and_reject_outside_spans() {
        let cell: AxesGridCell =
            serde_json::from_str(r#"{"rows":2,"columns":3,"index":5}"#).unwrap();
        assert_eq!(cell, AxesGridCell::single(2, 3, 5));
        assert!(cell.is_valid());
        assert!(!AxesGridCell {
            row_span: 2,
            ..cell
        }
        .is_valid());
        assert!(!AxesGridCell {
            column_span: 0,
            ..cell
        }
        .is_valid());
        assert_eq!(AxesGridCell::from_slots(0, 3, [0]), None);
        assert_eq!(AxesGridCell::from_slots(2, 3, []), None);
        assert_eq!(AxesGridCell::from_slots(2, 3, [6]), None);
        assert_eq!(AxesGridCell::from_slots(u32::MAX, 2, [0]), None);
    }
}
