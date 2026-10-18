use super::host_logical_buffer::HostLogicalBuffer;
use super::{HostShapeInput, SharedShape};

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalArray {
    // Payload and immutable shape each retain their own allocation claim.
    pub shape: SharedShape,
    pub data: HostLogicalBuffer,
}

impl LogicalArray {
    pub fn new(data: Vec<u8>, shape: impl Into<HostShapeInput>) -> Result<Self, String> {
        let shape = shape.into();
        let dimensions = match &shape {
            HostShapeInput::Owned(dimensions) => dimensions.as_slice(),
            HostShapeInput::Shared(dimensions) => dimensions.as_slice(),
        };
        let expected = dimensions
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or_else(|| format!("LogicalArray shape {dimensions:?} overflows usize"))?;
        if data.len() != expected {
            return Err(format!(
                "LogicalArray data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                dimensions,
                expected
            ));
        }
        Self::from_host_buffer(HostLogicalBuffer::new(data), shape)
    }

    pub fn from_host_buffer(
        data: HostLogicalBuffer,
        shape: impl Into<HostShapeInput>,
    ) -> Result<Self, String> {
        let shape = match shape.into() {
            HostShapeInput::Owned(shape) => SharedShape::new_unmanaged(shape),
            HostShapeInput::Shared(shape) => shape,
        };
        Self::from_shared_host_buffer(data, shape)
    }

    /// Reuse an existing immutable shape owner when only logical storage
    /// changes. This avoids materializing a second dimensions vector.
    pub fn from_shared_host_buffer(
        data: HostLogicalBuffer,
        shape: SharedShape,
    ) -> Result<Self, String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        if expected != Some(data.len()) {
            return Err(format!(
                "LogicalArray data length {} doesn't match shape {:?}",
                data.len(),
                shape
            ));
        }
        Ok(Self { data, shape })
    }

    /// Installs a current-operation shape owner after validating its extent.
    pub fn reshape_with_shared_shape(&mut self, shape: SharedShape) -> Result<(), String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        if expected != Some(self.data.len()) {
            return Err(format!(
                "LogicalArray data length {} doesn't match shape {:?}",
                self.data.len(),
                shape
            ));
        }
        self.shape = shape;
        Ok(())
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        LogicalArray {
            data: HostLogicalBuffer::new(vec![0u8; expected]),
            shape: SharedShape::new_unmanaged(shape),
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shared_shape(&self) -> SharedShape {
        self.shape.clone()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
