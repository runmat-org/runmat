use super::host_logical_buffer::HostLogicalBuffer;

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalArray {
    pub data: HostLogicalBuffer,
    pub shape: Vec<usize>,
}

impl LogicalArray {
    pub fn new(data: Vec<u8>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "LogicalArray data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        Ok(LogicalArray {
            data: HostLogicalBuffer::new(data),
            shape,
        })
    }

    pub fn from_host_buffer(data: HostLogicalBuffer, shape: Vec<usize>) -> Result<Self, String> {
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
    pub fn zeros(shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        LogicalArray {
            data: HostLogicalBuffer::new(vec![0u8; expected]),
            shape,
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
