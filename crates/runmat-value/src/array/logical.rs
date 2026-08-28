#[derive(Debug, Clone, PartialEq)]
pub struct LogicalArray {
    pub data: Vec<u8>, // 0 or 1 values; compact bitset can come later
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
        // Normalize to 0/1
        let mut d = data;
        for v in &mut d {
            *v = if *v != 0 { 1 } else { 0 };
        }
        Ok(LogicalArray { data: d, shape })
    }
    pub fn zeros(shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        LogicalArray {
            data: vec![0u8; expected],
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
