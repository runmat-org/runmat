use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct CharArray {
    pub(crate) data: ExactArrayStorage<char>,
    /// Full MATLAB-visible shape vector. Character payloads retain their
    /// historical row-major layout within each 2-D page.
    pub shape: SharedShape,
    /// Cached row count for 2-D interop; equals `shape[0]` when present.
    pub rows: usize,
    /// Cached column count for 2-D interop; equals `shape[1]` when present,
    /// otherwise 1 (or 0 for an empty shape).
    pub cols: usize,
}

impl CharArray {
    /// Authenticate cached two-dimensional geometry and every page against
    /// the canonical shape before a consumer reuses or transforms the owner.
    pub fn validate_layout(&self) -> Result<(), String> {
        let expected = total_len(&self.shape)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != self.data.len() || shape_rows_cols(&self.shape) != (self.rows, self.cols) {
            return Err("Character shape, cache, and payload disagree".into());
        }
        Ok(())
    }

    pub fn new_row(s: &str) -> Self {
        let data: Vec<char> = s.chars().collect();
        let cols = data.len();
        CharArray {
            data: ExactArrayStorage::shared_unbounded(data),
            shape: SharedShape::new_unmanaged(vec![1, cols]),
            rows: 1,
            cols,
        }
    }
    pub fn new(data: Vec<char>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new_with_shape(data, vec![rows, cols])
    }
    pub fn new_with_shape(data: Vec<char>, shape: Vec<usize>) -> Result<Self, String> {
        let expected = total_len(&shape)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Char data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = shape_rows_cols(&shape);
        Ok(CharArray {
            data: ExactArrayStorage::shared_unbounded(data),
            shape: SharedShape::new_unmanaged(shape),
            rows,
            cols,
        })
    }

    /// Publish already admitted payload and shape without a second backing.
    /// The producer must have reserved both owners before its first effect.
    pub fn from_admitted(
        data: ExactArrayStorage<char>,
        shape: SharedShape,
    ) -> Result<Self, String> {
        if !data.is_exact() || !shape.is_admitted() {
            return Err("Character payload and shape require exact admission".into());
        }
        let expected = total_len(&shape)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err("Character payload length does not match admitted shape".into());
        }
        let (rows, cols) = shape_rows_cols(&shape);
        Ok(Self {
            data,
            shape,
            rows,
            cols,
        })
    }

    pub fn data(&self) -> &[char] {
        self.data.as_slice()
    }
    pub fn data_owner(&self) -> &ExactArrayStorage<char> {
        &self.data
    }
    pub fn into_parts(self) -> (ExactArrayStorage<char>, SharedShape, usize, usize) {
        (self.data, self.shape, self.rows, self.cols)
    }
    pub fn make_data_unique_unbounded(&mut self) -> Result<(), crate::ExactArrayMutationError> {
        self.data.make_unique_unbounded()
    }
    /// Publish a borrowed-owner transformation without copying characters or
    /// dimensions. Ordinary unpublished vectors acquire their original header.
    pub fn from_shared_storage(
        data: ExactArrayStorage<char>,
        shape: SharedShape,
    ) -> Result<Self, String> {
        if total_len(&shape) != Some(data.len()) {
            return Err("Character payload length does not match shape".into());
        }
        let data = if data.is_shallow_clone() {
            data
        } else {
            let values = data
                .try_into_owned()
                .expect("unpublished character slots are unique")
                .try_into_unbounded_vec()
                .expect("unpublished character slots have no claim");
            if values.capacity() == 0 {
                ExactArrayStorage::empty()
            } else {
                ExactArrayStorage::shared_unbounded(values)
            }
        };
        let (rows, cols) = shape_rows_cols(&shape);
        Ok(Self {
            data,
            shape,
            rows,
            cols,
        })
    }

    /// Mutate only a unique, fixed-length owner. Exact aliases refuse.
    pub fn try_data_mut_unique(&mut self) -> Option<&mut [char]> {
        self.data.try_as_mut_slice_unique()
    }

    pub fn into_owned_data(self) -> Result<OwnedExactArrayValues<char>, Self> {
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
    pub fn from_column_major(data: Vec<char>, shape: Vec<usize>) -> Result<Self, String> {
        let normalized = match shape.as_slice() {
            [] => vec![0, 0],
            [length] => vec![1, *length],
            _ => shape,
        };
        let expected = total_len(&normalized)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Char data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                normalized,
                expected
            ));
        }
        let rows = normalized[0];
        let cols = normalized[1];
        let pages = total_len(&normalized[2..])
            .ok_or_else(|| "Char page shape exceeds platform limits".to_string())?;
        let pages = if normalized.len() <= 2 { 1 } else { pages };
        let mut row_major = Vec::with_capacity(data.len());
        for page in 0..pages {
            let page_offset = page * rows * cols;
            for row in 0..rows {
                for col in 0..cols {
                    row_major.push(data[page_offset + row + col * rows]);
                }
            }
        }
        Self::new_with_shape(row_major, normalized)
    }
    pub fn to_column_major(&self) -> Vec<char> {
        self.iter_column_major().collect()
    }

    /// Visit MATLAB linear order without materializing a second character
    /// payload. The array retains its backing for the iterator's lifetime.
    pub fn iter_column_major(&self) -> impl ExactSizeIterator<Item = char> + '_ {
        let page_len = self.rows * self.cols;
        (0..self.data.len()).map(move |linear| {
            let page = linear / page_len.max(1);
            let within_page = linear % page_len.max(1);
            let row = within_page % self.rows.max(1);
            let col = within_page / self.rows.max(1);
            self.data[page * page_len + row * self.cols + col]
        })
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the character contents when this value is a MATLAB character
    /// row vector. `Display` intentionally renders arrays for the console and
    /// therefore includes layout whitespace; semantic string conversions must
    /// use this method instead.
    pub fn row_string(&self) -> Option<String> {
        (self.rows == 1).then(|| self.data.iter().collect())
    }
}
