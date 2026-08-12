#[derive(Debug, Clone, PartialEq)]
pub struct StringArray {
    pub data: Vec<String>,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
}

impl StringArray {
    pub fn new(data: Vec<String>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "StringArray data length {} doesn't match shape {:?} ({} elements)",
                data.len(),
                shape,
                expected
            ));
        }
        let (rows, cols) = if shape.len() >= 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            (0, 0)
        };
        Ok(StringArray {
            data,
            shape,
            rows,
            cols,
        })
    }
    pub fn new_2d(data: Vec<String>, rows: usize, cols: usize) -> Result<Self, String> {
        Self::new(data, vec![rows, cols])
    }
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }
}
