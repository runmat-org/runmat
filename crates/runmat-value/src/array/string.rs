use crate::{
    ExactArrayStorage, ExactHostNumericAllocationError, SharedShape, StringElement, Utf8Text,
};

use super::shape_rows_cols;

fn string_element_count(shape: &[usize]) -> Option<usize> {
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
}

/// Column-major string values. The collection, shape, and every nonempty
/// UTF-8 payload have independent physical owners under finite admission.
#[derive(Debug, Clone, PartialEq)]
pub struct StringArray {
    pub(crate) data: ExactArrayStorage<StringElement>,
    pub shape: SharedShape,
    pub rows: usize,
    pub cols: usize,
}

impl StringArray {
    /// Authenticate the cached two-dimensional view and every page against
    /// the StringArray's canonical column-major shape and payload count.
    pub fn validate_layout(&self) -> Result<(), String> {
        let expected = string_element_count(&self.shape)
            .ok_or_else(|| "StringArray shape exceeds platform limits".to_string())?;
        if expected != self.data.len() || shape_rows_cols(&self.shape) != (self.rows, self.cols) {
            return Err("StringArray shape, cache, and payload disagree".into());
        }
        Ok(())
    }

    /// Ordinary construction moves each String's byte allocation into its
    /// immutable text owner. Only the outer owner vector is newly allocated.
    pub fn new(data: Vec<String>, shape: Vec<usize>) -> Result<Self, String> {
        Self::validate_length(data.len(), &shape)?;
        Ok(Self::from_unbounded_parts(
            data.into_iter().map(StringElement::from).collect(),
            shape,
        ))
    }

    /// Ordinary construction from existing text owners. Text byte payloads
    /// move unchanged into present elements; admitted children keep their
    /// claims. The typed element collection and shape are not admitted here.
    pub fn from_texts_unbounded(data: Vec<Utf8Text>, shape: Vec<usize>) -> Result<Self, String> {
        Self::validate_length(data.len(), &shape)?;
        Ok(Self::from_unbounded_parts(
            data.into_iter().map(StringElement::Text).collect(),
            shape,
        ))
    }

    /// Move typed elements and their existing text owners without flattening
    /// missingness or copying byte payloads.
    pub fn from_elements_unbounded(
        data: Vec<StringElement>,
        shape: Vec<usize>,
    ) -> Result<Self, String> {
        Self::validate_length(data.len(), &shape)?;
        Ok(Self::from_unbounded_parts(data, shape))
    }

    fn validate_length(length: usize, shape: &[usize]) -> Result<(), String> {
        let expected = string_element_count(shape)
            .ok_or_else(|| "StringArray shape exceeds platform limits".to_string())?;
        if length != expected {
            return Err(format!(
                "StringArray data length {} doesn't match shape {:?} ({} elements)",
                length, shape, expected
            ));
        }
        Ok(())
    }

    fn from_unbounded_parts(data: Vec<StringElement>, shape: Vec<usize>) -> Self {
        Self::from_validated_storage(
            ExactArrayStorage::shared_unbounded(data),
            SharedShape::new_unmanaged(shape),
        )
    }

    fn from_validated_storage(data: ExactArrayStorage<StringElement>, shape: SharedShape) -> Self {
        let (rows, cols) = shape_rows_cols(&shape);
        let data = if data.is_shallow_clone() {
            data
        } else {
            let values = data
                .try_into_owned()
                .expect("unpublished string slots are unique")
                .try_into_unbounded_vec()
                .expect("unpublished string slots have no claim");
            if values.capacity() == 0 {
                ExactArrayStorage::empty()
            } else {
                ExactArrayStorage::shared_unbounded(values)
            }
        };
        Self {
            data,
            shape,
            rows,
            cols,
        }
    }

    /// Validate and move existing collection and shape owners without copying
    /// children or allocating. This does not certify finite admission; callers
    /// publishing newly admitted outputs use `from_admitted` instead.
    pub fn from_shared_storage(
        data: ExactArrayStorage<StringElement>,
        shape: SharedShape,
    ) -> Result<Self, String> {
        Self::validate_length(data.len(), &shape)?;
        Ok(Self::from_validated_storage(data, shape))
    }

    /// Publish the original collection and child byte allocations without
    /// copying or re-quoting; all three owner layers must already be admitted.
    pub fn from_admitted(
        data: ExactArrayStorage<StringElement>,
        shape: SharedShape,
    ) -> Result<Self, String> {
        if !data.is_exact() || !shape.is_admitted() {
            return Err("StringArray collection and shape require exact admission".into());
        }
        if data
            .iter()
            .any(|text| text.capacity() > 0 && !text.has_retained_claim())
        {
            return Err("StringArray UTF-8 payload requires exact admission".into());
        }
        let expected = string_element_count(&shape)
            .ok_or_else(|| "StringArray shape exceeds platform limits".to_string())?;
        if data.len() != expected {
            return Err("StringArray data length does not match admitted shape".into());
        }
        Ok(Self::from_validated_storage(data, shape))
    }

    /// Retained outer, new shape, and newly created UTF-8 capacity quote.
    /// Existing admitted aliases are already claimed and must not be charged
    /// again. Excludes permit claim-control and operation scratch.
    pub fn requested_bytes_for_new_dimensions(
        element_count: usize,
        rank: usize,
        new_text_capacities: impl IntoIterator<Item = usize>,
    ) -> Result<usize, ExactHostNumericAllocationError> {
        let mut bytes = ExactArrayStorage::<StringElement>::requested_bytes(element_count)?
            .checked_add(SharedShape::requested_bytes_for_rank(rank)?)
            .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        for capacity in new_text_capacities {
            bytes = bytes
                .checked_add(Utf8Text::requested_bytes_for_capacity(capacity)?)
                .ok_or(ExactHostNumericAllocationError::LayoutOverflow)?;
        }
        Ok(bytes)
    }

    pub fn new_2d(data: Vec<String>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }

    pub fn data(&self) -> &[StringElement] {
        self.data.as_slice()
    }
    pub fn data_owner(&self) -> &ExactArrayStorage<StringElement> {
        &self.data
    }
    pub fn into_parts(self) -> (ExactArrayStorage<StringElement>, SharedShape, usize, usize) {
        (self.data, self.shape, self.rows, self.cols)
    }
    pub fn make_data_unique_unbounded(&mut self) -> Result<(), crate::ExactArrayMutationError> {
        self.data.make_unique_unbounded()
    }

    /// Grow ordinary storage with empty strings, publishing collection and
    /// shape together. Admitted arrays require a newly admitted destination.
    pub fn grow_unbounded(&mut self, shape: Vec<usize>) -> Result<(), String> {
        let count = string_element_count(&shape)
            .ok_or_else(|| "StringArray shape exceeds platform limits".to_string())?;
        if count < self.data.len() {
            return Err("StringArray growth cannot remove elements".into());
        }
        if self.data.is_exact() || self.shape.is_admitted() {
            return Err("StringArray growth requires a new admitted destination".into());
        }
        self.data
            .make_unique_unbounded()
            .map_err(|e| e.to_string())?;
        let shape = SharedShape::new_unmanaged(shape);
        while self.data.len() < count {
            self.data
                .try_push_unbounded(StringElement::default())
                .map_err(|_| "StringArray growth requires unique ordinary storage".to_string())?;
        }
        let replacement = Self::from_validated_storage(std::mem::take(&mut self.data), shape);
        *self = replacement;
        Ok(())
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Same-cardinality mutation is safe only when the collection is unique.
    /// New strings still require independent byte admission on finite routes.
    pub fn try_data_mut_unique(&mut self) -> Option<&mut [StringElement]> {
        self.data.try_as_mut_slice_unique()
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }
}
