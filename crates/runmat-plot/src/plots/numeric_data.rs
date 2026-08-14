use runmat_builtins::NumericStorage;

/// Authoritative host-side numeric data retained by a graphics object.
///
/// Rendering may explicitly materialize this storage into a floating geometry
/// domain, but graphics properties reconstruct their values from this native
/// payload so class, exact integer values, and array shape are not lost.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericPlotData {
    storage: NumericStorage,
    shape: Vec<usize>,
}

impl NumericPlotData {
    pub fn new(storage: NumericStorage, shape: Vec<usize>) -> Result<Self, String> {
        storage.validate_shape(&shape)?;
        Ok(Self { storage, shape })
    }

    pub fn from_f64(values: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        Self::new(NumericStorage::F64(values), shape)
    }

    pub fn storage(&self) -> &NumericStorage {
        &self.storage
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Explicit renderer-domain materialization. Graphics properties must use
    /// `storage` directly instead of routing back through this conversion.
    pub fn materialize_f64(&self) -> Vec<f64> {
        self.storage.materialize_f64()
    }

    pub fn estimated_byte_len(&self) -> usize {
        self.storage.checked_byte_len().unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_plot_data_keeps_wide_integer_storage_authoritative() {
        let wide = 9_007_199_254_740_993_u64;
        let data = NumericPlotData::new(NumericStorage::U64(vec![wide]), vec![1, 1]).unwrap();

        assert_eq!(data.storage(), &NumericStorage::U64(vec![wide]));
        assert_eq!(data.shape(), &[1, 1]);
    }
}
