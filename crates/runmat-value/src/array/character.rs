use super::*;
mod admission;
use admission::character_len;
pub use admission::CharArrayAdmission;

/// One MATLAB-compatible UTF-16 character code unit.
///
/// This is deliberately not a Unicode scalar value. Surrogate code units are
/// retained exactly so character-array geometry, indexing, persistence, and
/// foreign interchange all observe the same payload.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct Utf16CodeUnit(u16);

const _: () = {
    assert!(std::mem::size_of::<Utf16CodeUnit>() == std::mem::size_of::<u16>());
    assert!(std::mem::align_of::<Utf16CodeUnit>() == std::mem::align_of::<u16>());
};

impl Utf16CodeUnit {
    pub const fn from_raw(value: u16) -> Self {
        Self(value)
    }

    pub const fn raw(self) -> u16 {
        self.0
    }

    pub const fn is_surrogate(self) -> bool {
        matches!(self.0, 0xD800..=0xDFFF)
    }

    pub const fn is_ascii(self) -> bool {
        self.0 <= 0x7F
    }

    pub fn is_ascii_whitespace(self) -> bool {
        u8::try_from(self.0).is_ok_and(|value| value.is_ascii_whitespace())
    }

    pub fn is_whitespace(self) -> bool {
        char::from_u32(self.0 as u32).is_some_and(char::is_whitespace)
    }
}

impl From<u16> for Utf16CodeUnit {
    fn from(value: u16) -> Self {
        Self::from_raw(value)
    }
}

impl From<Utf16CodeUnit> for u16 {
    fn from(value: Utf16CodeUnit) -> Self {
        value.raw()
    }
}

impl From<Utf16CodeUnit> for u32 {
    fn from(value: Utf16CodeUnit) -> Self {
        value.raw() as u32
    }
}

impl fmt::Display for Utf16CodeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match char::from_u32(self.raw() as u32) {
            Some(value) if !self.is_surrogate() => write!(f, "{value}"),
            _ => write!(f, "\\u{:04X}", self.raw()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Utf16DecodeError {
    pub index: usize,
    pub unpaired_surrogate: u16,
}

impl fmt::Display for Utf16DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unpaired UTF-16 surrogate 0x{:04X} at code-unit index {}",
            self.unpaired_surrogate, self.index
        )
    }
}

impl std::error::Error for Utf16DecodeError {}

pub fn encode_utf16_code_units(text: &str) -> Vec<Utf16CodeUnit> {
    text.encode_utf16().map(Utf16CodeUnit::from_raw).collect()
}

pub fn decode_utf16_code_units(units: &[Utf16CodeUnit]) -> Result<String, Utf16DecodeError> {
    let mut text = String::new();
    let mut index = 0;
    while index < units.len() {
        let first = units[index].raw();
        if !(0xD800..=0xDFFF).contains(&first) {
            // Every non-surrogate u16 is a Unicode scalar value.
            text.push(char::from_u32(first as u32).expect("non-surrogate u16 is a scalar"));
            index += 1;
            continue;
        }

        if !(0xD800..=0xDBFF).contains(&first) {
            return Err(Utf16DecodeError {
                index,
                unpaired_surrogate: first,
            });
        }
        let Some(second) = units.get(index + 1).map(|unit| unit.raw()) else {
            return Err(Utf16DecodeError {
                index,
                unpaired_surrogate: first,
            });
        };
        if !(0xDC00..=0xDFFF).contains(&second) {
            return Err(Utf16DecodeError {
                index,
                unpaired_surrogate: first,
            });
        }
        let scalar = 0x1_0000 + (((first as u32 - 0xD800) << 10) | (second as u32 - 0xDC00));
        text.push(char::from_u32(scalar).expect("validated UTF-16 surrogate pair is a scalar"));
        index += 2;
    }
    Ok(text)
}

#[derive(Debug, Clone, PartialEq)]
pub struct CharArray {
    pub(crate) data: ExactArrayStorage<Utf16CodeUnit>,
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
        let expected = character_len(&self.shape)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != self.data.len() || shape_rows_cols(&self.shape) != (self.rows, self.cols) {
            return Err("Character shape, cache, and payload disagree".into());
        }
        Ok(())
    }

    pub fn new_row(s: &str) -> Self {
        let data = encode_utf16_code_units(s);
        let cols = data.len();
        CharArray {
            data: ExactArrayStorage::shared_unbounded(data),
            shape: SharedShape::new_unmanaged(vec![1, cols]),
            rows: 1,
            cols,
        }
    }
    pub fn from_unicode_scalars(data: Vec<char>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::from_unicode_scalars_with_shape(data, vec![rows, cols])
    }
    /// Construct from Unicode scalar values while interpreting `shape` in
    /// UTF-16 code units. Supplementary scalars therefore occupy two slots.
    pub fn from_unicode_scalars_with_shape(
        data: Vec<char>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let mut units = Vec::new();
        for value in data {
            let mut encoded = [0_u16; 2];
            units.extend(
                value
                    .encode_utf16(&mut encoded)
                    .iter()
                    .copied()
                    .map(Utf16CodeUnit::from_raw),
            );
        }
        Self::from_code_units_with_shape(units, shape)
    }

    /// Construct from the canonical raw character representation.
    pub fn from_code_units(
        data: Vec<Utf16CodeUnit>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        Self::from_code_units_with_shape(data, vec![rows, cols])
    }

    /// Construct from raw UTF-16 code units without decoding or normalizing
    /// surrogate pairs.
    pub fn from_code_units_with_shape(
        data: Vec<Utf16CodeUnit>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        let expected = character_len(&shape)
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
        data: ExactArrayStorage<Utf16CodeUnit>,
        shape: SharedShape,
    ) -> Result<Self, String> {
        if !data.is_exact() || !shape.is_admitted() {
            return Err("Character payload and shape require exact admission".into());
        }
        let expected = character_len(&shape)
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

    pub fn data(&self) -> &[Utf16CodeUnit] {
        self.data.as_slice()
    }
    pub fn data_owner(&self) -> &ExactArrayStorage<Utf16CodeUnit> {
        &self.data
    }
    pub fn into_parts(self) -> (ExactArrayStorage<Utf16CodeUnit>, SharedShape, usize, usize) {
        (self.data, self.shape, self.rows, self.cols)
    }
    pub fn make_data_unique_unbounded(&mut self) -> Result<(), crate::ExactArrayMutationError> {
        self.data.make_unique_unbounded()
    }
    /// Publish a borrowed-owner transformation without copying characters or
    /// dimensions. Ordinary unpublished vectors acquire their original header.
    pub fn from_shared_storage(
        data: ExactArrayStorage<Utf16CodeUnit>,
        shape: SharedShape,
    ) -> Result<Self, String> {
        if character_len(&shape) != Some(data.len()) {
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
    pub fn try_data_mut_unique(&mut self) -> Option<&mut [Utf16CodeUnit]> {
        self.data.try_as_mut_slice_unique()
    }

    pub fn into_owned_data(self) -> Result<OwnedExactArrayValues<Utf16CodeUnit>, Self> {
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
    pub fn from_column_major(
        data: Vec<Utf16CodeUnit>,
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
        let finish = |data, shape| match shape {
            HostShapeInput::Owned(shape) => Self::from_code_units_with_shape(data, shape),
            HostShapeInput::Shared(shape) => {
                Self::from_shared_storage(ExactArrayStorage::shared_unbounded(data), shape)
            }
        };
        let expected = character_len(normalized)
            .ok_or_else(|| "Char data shape exceeds platform limits".to_string())?;
        if expected != data.len() {
            return Err(format!(
                "Char data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                normalized,
                expected
            ));
        }
        if expected == 0 {
            return finish(data, shape);
        }
        let rows = normalized[0];
        let cols = normalized[1];
        // Each vector page has the same physical order in both layouts.
        if rows == 1 || cols == 1 {
            return finish(data, shape);
        }
        let pages = character_len(&normalized[2..])
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
        finish(row_major, shape)
    }
    pub fn to_column_major(&self) -> Vec<Utf16CodeUnit> {
        self.iter_column_major().collect()
    }

    /// Visit MATLAB linear order without materializing a second character
    /// payload. The array retains its backing for the iterator's lifetime.
    pub fn iter_column_major(&self) -> impl ExactSizeIterator<Item = Utf16CodeUnit> + '_ {
        (0..self.data.len()).map(move |linear| {
            self.column_major_value_at(linear)
                .expect("valid character layout")
        })
    }

    /// Address one MATLAB-linear code unit without a column-major mirror.
    pub fn column_major_value_at(&self, linear: usize) -> Option<Utf16CodeUnit> {
        let physical = Self::row_major_position(self.rows, self.cols, linear)?;
        self.data.get(physical).copied()
    }

    /// The same zero-aware code-unit cardinality used by character
    /// construction and admission, including an empty-rank empty payload.
    pub fn code_unit_count_for_shape(shape: &[usize]) -> Option<usize> {
        character_len(shape)
    }

    /// Canonical page-preserving layout map, also usable while a producer
    /// exclusively borrows a final character backing for indexed mutation.
    pub fn row_major_position(rows: usize, cols: usize, linear: usize) -> Option<usize> {
        let page_len = rows.checked_mul(cols)?;
        if page_len == 0 {
            return None;
        }
        let page = linear / page_len;
        let within = linear % page_len;
        page.checked_mul(page_len)?
            .checked_add((within % rows).checked_mul(cols)?)?
            .checked_add(within / rows)
    }

    /// Inverse of row_major_position for direct final row-major fill.
    pub fn column_major_position(rows: usize, cols: usize, physical: usize) -> Option<usize> {
        let page_len = rows.checked_mul(cols)?;
        if page_len == 0 {
            return None;
        }
        let page = physical / page_len;
        let within = physical % page_len;
        page.checked_mul(page_len)?
            .checked_add(within / cols)?
            .checked_add((within % cols).checked_mul(rows)?)
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the character contents when this value is a MATLAB character
    /// row vector. `Display` intentionally renders arrays for the console and
    /// therefore includes layout whitespace; semantic string conversions must
    /// use this method instead.
    pub fn try_row_string(&self) -> Result<Option<String>, Utf16DecodeError> {
        if self.rows != 1 {
            return Ok(None);
        }
        decode_utf16_code_units(self.data()).map(Some)
    }

    /// Decode a row when it is valid Unicode text. Non-row geometry and
    /// malformed UTF-16 both refuse conversion; callers that surface causes
    /// should use [`Self::try_row_string`].
    pub fn row_string(&self) -> Option<String> {
        self.try_row_string().ok().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_rows_use_utf16_code_unit_geometry() {
        let value = CharArray::new_row("A\u{1F642}B");
        assert_eq!(value.shape(), &[1, 4]);
        assert_eq!(
            value
                .data()
                .iter()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>(),
            vec![0x0041, 0xD83D, 0xDE42, 0x0042]
        );
        assert_eq!(
            value.try_row_string().unwrap().as_deref(),
            Some("A\u{1F642}B")
        );
    }

    #[test]
    fn raw_code_units_preserve_unpaired_surrogates_without_text_normalization() {
        let value = CharArray::from_code_units_with_shape(
            vec![
                Utf16CodeUnit::from_raw(0x0061),
                Utf16CodeUnit::from_raw(0xD800),
            ],
            vec![1, 2],
        )
        .unwrap();
        assert_eq!(value.data()[1].raw(), 0xD800);
        let column_major = value.to_column_major();
        let original_data = column_major.as_ptr();
        let rebuilt = CharArray::from_column_major(column_major, value.shape.clone()).unwrap();
        assert_eq!(rebuilt, value);
        assert_eq!(rebuilt.data().as_ptr(), original_data);
        assert_eq!(rebuilt.shape.as_ptr(), value.shape.as_ptr());
        assert_eq!(
            value.try_row_string(),
            Err(Utf16DecodeError {
                index: 1,
                unpaired_surrogate: 0xD800,
            })
        );
        for shape in [vec![usize::MAX, 2, 0], vec![0, usize::MAX, 7], vec![]] {
            let value = CharArray::from_code_units_with_shape(Vec::new(), shape.clone()).unwrap();
            assert_eq!(value.shape(), shape);
            assert!(value.data().is_empty());
            assert_eq!(value.iter_column_major().len(), 0);
        }
        assert!(CharArray::from_code_units_with_shape(Vec::new(), vec![usize::MAX, 2]).is_err());
        let empty = CharArray::from_column_major(Vec::new(), vec![0, 2, usize::MAX]).unwrap();
        assert_eq!(empty.shape(), &[0, 2, usize::MAX]);
    }

    #[test]
    fn decode_error_reports_the_exact_code_unit_after_a_surrogate_pair() {
        let units = vec![
            Utf16CodeUnit::from_raw(0xD83D),
            Utf16CodeUnit::from_raw(0xDE42),
            Utf16CodeUnit::from_raw(0xDC00),
        ];
        assert_eq!(
            decode_utf16_code_units(&units),
            Err(Utf16DecodeError {
                index: 2,
                unpaired_surrogate: 0xDC00,
            })
        );
    }

    #[test]
    fn scalar_constructor_shape_is_checked_in_code_units() {
        assert!(CharArray::from_unicode_scalars(vec!['\u{1F642}'], 1, 1).is_err());
        assert_eq!(
            CharArray::from_unicode_scalars(vec!['\u{1F642}'], 1, 2)
                .unwrap()
                .data()
                .len(),
            2
        );
    }
}
