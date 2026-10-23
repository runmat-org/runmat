use super::*;
use crate::{HostShapeInput, OutputListStorage, OwnedExactArrayValues, SharedShape};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CellArray {
    pub(crate) data: OutputListStorage,
    /// Full MATLAB-visible shape vector. Cell payloads retain their historical
    /// row-major layout within each 2-D page.
    pub shape: SharedShape,
    /// Cached row count for 2-D interop; equals `shape[0]` when present.
    pub rows: usize,
    /// Cached column count for 2-D interop; equals `shape[1]` when present, otherwise 1 (or 0 for empty).
    pub cols: usize,
}

impl CellArray {
    /// Authenticate the cached two-dimensional view and every row-major page
    /// against the canonical Cell shape and payload count.
    pub fn validate_layout(&self) -> Result<(), String> {
        let expected = total_len(&self.shape)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != self.data.len() || shape_rows_cols(&self.shape) != (self.rows, self.cols) {
            return Err("Cell shape, cache, and payload disagree".into());
        }
        Ok(())
    }

    pub fn new(data: Vec<Value>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new_with_shape(data, vec![rows, cols])
    }

    pub fn new_with_shape(data: Vec<Value>, shape: Vec<usize>) -> Result<Self, String> {
        let expected = total_len(&shape)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Cell data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        Ok(Self::from_validated_storage(
            OutputListStorage::shared_unbounded(data),
            SharedShape::new_unmanaged(shape),
        ))
    }

    /// Validate and move existing row-major collection and shape owners. No
    /// child or shape backing is copied, and no new admission is implied.
    pub fn from_shared_storage(
        data: OutputListStorage,
        shape: SharedShape,
    ) -> Result<Self, String> {
        let expected = total_len(&shape)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Cell data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        Ok(Self::from_validated_storage(data, shape))
    }

    fn from_validated_storage(data: OutputListStorage, shape: SharedShape) -> Self {
        let (rows, cols) = shape_rows_cols(&shape);
        Self {
            data: shared_original(data),
            shape,
            rows,
            cols,
        }
    }

    /// Publish separately admitted collection and shape owners without
    /// cloning children or allocating a replacement collection.
    pub fn from_admitted(data: OutputListStorage, shape: SharedShape) -> Result<Self, String> {
        if !data.is_exact() || !shape.is_admitted() {
            return Err("Cell payload and shape require exact admission".into());
        }
        let expected = total_len(&shape)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err("Cell payload length does not match admitted shape".into());
        }
        Ok(Self::from_validated_storage(data, shape))
    }

    pub fn data(&self) -> &[Value] {
        self.data.as_slice()
    }
    /// Borrow the immutable original physical owner, not a mutable Vec facade.
    pub fn data_owner(&self) -> &OutputListStorage {
        &self.data
    }
    /// Move the existing owners without dropping their retained claims or
    /// silently repairing deliberately malformed legacy cached dimensions.
    pub fn into_parts(self) -> (OutputListStorage, SharedShape, usize, usize) {
        (self.data, self.shape, self.rows, self.cols)
    }
    /// Explicit ordinary alias separation. Publish the new original shared
    /// owner before returning; subsequent Cell clones never clone slot Vecs.
    pub fn make_data_unique_unbounded(&mut self) -> Result<(), crate::ExactArrayMutationError> {
        if self.data.is_unique() {
            return Ok(());
        }
        self.data.make_unique_unbounded()?;
        self.data = shared_original(std::mem::take(&mut self.data));
        Ok(())
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// The caller may replace existing children only when no alias can
    /// observe this exact backing. Length and capacity remain unchanged.
    pub fn try_data_mut_unique(&mut self) -> Option<&mut [Value]> {
        self.data.try_as_mut_slice_unique()
    }

    pub fn into_owned_data(self) -> Result<OwnedExactArrayValues<Value>, Self> {
        let Self {
            data,
            shape,
            rows,
            cols,
        } = self;
        data.try_into_owned().map_err(|data| Self {
            data,
            shape,
            rows,
            cols,
        })
    }

    /// Append to a 1×N cell row while keeping its full shape and cached
    /// dimensions in sync. Reject a malformed or non-row array unchanged.
    pub fn push_row_element(&mut self, value: Value) -> Result<(), String> {
        if self.shape.as_slice() != [1, self.data.len()]
            || self.rows != 1
            || self.cols != self.data.len()
        {
            return Err("Cell array is not a valid row vector".into());
        }
        if self.data.is_exact() || self.shape.is_admitted() {
            return Err("Cell row growth requires a new admitted output".into());
        }
        let next_count = self
            .data
            .len()
            .checked_add(1)
            .ok_or_else(|| "Cell row length exceeds platform limits".to_string())?;
        // Ordinary aliases explicitly copy on growth; finite exact payloads
        // were rejected above and must be rebuilt under new admission.
        self.data
            .make_unique_unbounded()
            .map_err(|_| "Cell row growth requires ordinary storage".to_string())?;
        let next_shape = SharedShape::new_unmanaged(vec![1, next_count]);
        self.data
            .try_push_unbounded(value)
            .map_err(|_| "Cell row growth requires ordinary storage".to_string())?;
        self.data = shared_original(std::mem::take(&mut self.data));
        self.shape = next_shape;
        self.cols = next_count;
        Ok(())
    }

    pub fn from_column_major(
        data: Vec<Value>,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        let shape = match shape.into() {
            HostShapeInput::Owned(shape) => HostShapeInput::Owned(match shape.as_slice() {
                [] => vec![0, 0],
                [length] => vec![1, *length],
                _ => shape,
            }),
            HostShapeInput::Shared(shape) => match shape.as_slice() {
                [] => HostShapeInput::Owned(vec![0, 0]),
                [length] => HostShapeInput::Owned(vec![1, *length]),
                _ => HostShapeInput::Shared(shape),
            },
        };
        let normalized = match &shape {
            HostShapeInput::Owned(shape) => shape.as_slice(),
            HostShapeInput::Shared(shape) => shape.as_slice(),
        };
        let expected = total_len(normalized)
            .ok_or_else(|| "Cell data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Cell data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                normalized,
                expected
            ));
        }
        if expected == 0 {
            return match shape {
                HostShapeInput::Owned(shape) => Self::new_with_shape(data, shape),
                HostShapeInput::Shared(shape) => {
                    Self::from_shared_storage(OutputListStorage::shared_unbounded(data), shape)
                }
            };
        }
        let rows = normalized[0];
        let cols = normalized[1];
        let pages = if normalized.len() <= 2 {
            1
        } else {
            total_len(&normalized[2..])
                .ok_or_else(|| "Cell page shape exceeds platform limits".to_string())?
        };
        let mut source = data.into_iter().map(Some).collect::<Vec<_>>();
        let mut row_major = Vec::with_capacity(source.len());
        for page in 0..pages {
            let page_offset = page * rows * cols;
            for row in 0..rows {
                for col in 0..cols {
                    let index = page_offset + row + col * rows;
                    let value = source
                        .get_mut(index)
                        .and_then(Option::take)
                        .ok_or_else(|| "cell storage does not match its shape".to_string())?;
                    row_major.push(value);
                }
            }
        }
        match shape {
            HostShapeInput::Owned(shape) => Self::new_with_shape(row_major, shape),
            HostShapeInput::Shared(shape) => {
                Self::from_shared_storage(OutputListStorage::shared_unbounded(row_major), shape)
            }
        }
    }

    pub fn to_column_major(&self) -> Vec<Value> {
        self.iter_column_major().cloned().collect()
    }

    /// Move unique ordinary values into MATLAB-visible linear order. Ordinary
    /// aliases copy on write; admitted backing requires an admitted destination
    /// instead of shedding its allocation claim into a bare vector.
    pub fn into_column_major(self) -> Result<Vec<Value>, String> {
        let Self {
            mut data,
            rows,
            cols,
            ..
        } = self;
        data.make_unique_unbounded()
            .map_err(|_| "Admitted cell storage requires an admitted destination".to_string())?;
        let source = data
            .try_into_owned()
            .map_err(|_| "Shared cell storage cannot be moved into an ordinary vector".to_string())?
            .try_into_unbounded_vec()
            .map_err(|_| "Admitted cell storage cannot shed its allocation claim".to_string())?;
        if source.is_empty() || rows <= 1 || cols <= 1 {
            return Ok(source);
        }
        let page_len = rows * cols;
        let mut source = source.into_iter().map(Some).collect::<Vec<_>>();
        let mut ordered = Vec::with_capacity(source.len());
        for page_offset in (0..source.len()).step_by(page_len) {
            for column in 0..cols {
                for row in 0..rows {
                    let index = page_offset + row * cols + column;
                    let value = source
                        .get_mut(index)
                        .and_then(Option::take)
                        .ok_or_else(|| "cell storage does not match its shape".to_string())?;
                    ordered.push(value);
                }
            }
        }
        Ok(ordered)
    }

    /// Borrow one MATLAB-visible linear element in O(1), without cloning or
    /// materializing a reordered buffer. Cell storage is row-major within each
    /// 2-D page; higher-dimensional page order is unchanged.
    pub fn get_column_major(&self, linear: usize) -> Option<&Value> {
        if linear >= self.data.len() {
            return None;
        }
        let page_len = self.rows.checked_mul(self.cols)?;
        let page = linear / page_len.max(1);
        let within_page = linear % page_len.max(1);
        let row = within_page % self.rows.max(1);
        let col = within_page / self.rows.max(1);
        let physical = page
            .checked_mul(page_len)?
            .checked_add(row.checked_mul(self.cols)?)?
            .checked_add(col)?;
        self.data.as_slice().get(physical)
    }

    /// Iterate in MATLAB-visible linear order without cloning cell payloads.
    /// For random access, use `get_column_major` rather than repeatedly calling
    /// `nth`: a mapped iterator must evaluate its skipped elements.
    pub fn iter_column_major(&self) -> impl Iterator<Item = &Value> {
        (0..self.data.len()).map(move |linear| {
            self.get_column_major(linear)
                .expect("cell storage does not match cached dimensions")
        })
    }

    pub fn get(&self, row: usize, col: usize) -> Result<Value, String> {
        if row >= self.rows || col >= self.cols {
            return Err(format!(
                "Cell index ({row}, {col}) out of bounds for {}x{} cell array",
                self.rows, self.cols
            ));
        }
        Ok(self.data[row * self.cols + col].clone())
    }
}

fn shared_original(data: OutputListStorage) -> OutputListStorage {
    if data.is_shallow_clone() {
        return data;
    }
    let values = data
        .try_into_owned()
        .expect("unpublished Cell slots are unique")
        .try_into_unbounded_vec()
        .expect("unpublished Cell slots have no physical claim");
    if values.capacity() == 0 {
        OutputListStorage::empty()
    } else {
        OutputListStorage::shared_unbounded(values)
    }
}

pub(crate) fn total_len(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

pub(crate) fn shape_rows_cols(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() {
        return (0, 0);
    }
    if shape.len() == 1 {
        return (1, shape[0]);
    }
    (shape[0], shape[1])
}

impl fmt::Display for CellArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        if self.shape.len() > 2 {
            return write!(f, "{} cell array", dims.join("x"));
        }
        write!(f, "{}x{} cell array", self.rows, self.cols)?;
        if self.rows == 0 || self.cols == 0 {
            return Ok(());
        }
        for r in 0..self.rows {
            writeln!(f)?;
            write!(f, "  ")?;
            for c in 0..self.cols {
                if c > 0 {
                    write!(f, "  ")?;
                }
                let value = self.get(r, c).unwrap_or_else(|_| Value::Num(f64::NAN));
                write!(f, "{{{value}}}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_column_major_iteration_matches_owned_projection() {
        let expected = vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let array = CellArray::from_column_major(expected.clone(), vec![2, 2]).unwrap();
        let borrowed = array.iter_column_major().cloned().collect::<Vec<_>>();
        assert_eq!(borrowed, array.to_column_major());
        assert_eq!(borrowed, expected);
        let rebuilt = CellArray::from_column_major(expected.clone(), array.shape.clone()).unwrap();
        assert_eq!(rebuilt, array);
        assert_eq!(rebuilt.shape.as_ptr(), array.shape.as_ptr());
        let shape = SharedShape::new_unmanaged(vec![usize::MAX, 0]);
        let empty = CellArray::from_column_major(Vec::new(), shape.clone()).unwrap();
        assert_eq!(empty.shape.as_ptr(), shape.as_ptr());
        assert!(empty.data().is_empty());
        for (index, value) in expected.iter().enumerate() {
            assert_eq!(array.get_column_major(index), Some(value));
        }
        assert_eq!(array.get_column_major(expected.len()), None);
        assert_eq!(array.get_column_major(usize::MAX), None);

        // Independent physical-storage oracle across pages, not a comparison
        // between two consumers of the same mapping implementation.
        let pages = CellArray::new_with_shape(
            (1..=8).map(|value| Value::Num(f64::from(value))).collect(),
            vec![2, 2, 2],
        )
        .unwrap();
        let alias = pages.clone();
        for (logical, physical) in [0, 2, 1, 3, 4, 6, 5, 7].into_iter().enumerate() {
            let borrowed = pages.get_column_major(logical).unwrap();
            assert!(std::ptr::eq(borrowed, &pages.data()[physical]));
            assert!(std::ptr::eq(
                borrowed,
                alias.get_column_major(logical).unwrap()
            ));
        }
        assert_eq!(pages.get_column_major(8), None);
    }

    #[test]
    fn borrowed_column_major_iteration_handles_empty_shape() {
        let array = CellArray::new(Vec::new(), 0, 0).unwrap();
        assert_eq!(array.iter_column_major().count(), 0);
        assert_eq!(array.get_column_major(0), None);
    }

    #[test]
    fn row_append_updates_full_shape_and_rejects_malformed_storage() {
        let mut row = CellArray::new(Vec::new(), 1, 0).unwrap();
        row.push_row_element(Value::Num(1.0)).unwrap();
        row.push_row_element(Value::Num(2.0)).unwrap();
        assert_eq!(row.shape, [1, 2]);
        assert_eq!((row.rows, row.cols), (1, 2));
        assert_eq!(row.data(), [Value::Num(1.0), Value::Num(2.0)]);

        row.shape = SharedShape::new_unmanaged(vec![1, 1]);
        assert!(row.push_row_element(Value::Num(3.0)).is_err());
        assert_eq!(row.data.len(), 2);
    }

    #[test]
    fn every_ordinary_publication_retains_original_shared_slots() {
        let values = vec![Value::Num(1.)];
        let pointer = values.as_ptr();
        let mut cell = CellArray::from_shared_storage(
            OutputListStorage::unbounded(values),
            SharedShape::new_unmanaged(vec![1, 1]),
        )
        .unwrap();
        assert_eq!(cell.data().as_ptr(), pointer);
        let alias = cell.clone();
        assert_eq!(alias.data().as_ptr(), pointer);
        cell.push_row_element(Value::Num(2.)).unwrap();
        assert_eq!(alias.data(), [Value::Num(1.)]);
        let next = cell.clone();
        assert_eq!(cell.data().as_ptr(), next.data().as_ptr());
        assert_eq!(cell.data(), [Value::Num(1.), Value::Num(2.)]);
    }

    #[test]
    fn owned_column_major_projection_moves_the_same_values() {
        let expected = vec![
            Value::Num(1.0),
            Value::Num(2.0),
            Value::Num(3.0),
            Value::Num(4.0),
        ];
        let array = CellArray::from_column_major(expected.clone(), vec![2, 2]).unwrap();
        let alias = array.clone();
        assert_eq!(array.data().as_ptr(), alias.data().as_ptr());
        assert_eq!(array.into_column_major().unwrap(), expected);
        assert_eq!(alias.into_column_major().unwrap(), expected);
    }
}
