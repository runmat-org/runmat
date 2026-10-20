//! Cell-only value traversal. Cells own their children and cannot form cycles;
//! reference-bearing objects remain terminal rather than introducing new semantics.
pub(super) struct CellLeaves<'a> {
    pending: Vec<&'a runmat_value::Value>,
}

impl<'a> CellLeaves<'a> {
    pub(super) fn new(value: &'a runmat_value::Value) -> Self {
        Self {
            pending: vec![value],
        }
    }
}

impl<'a> Iterator for CellLeaves<'a> {
    type Item = &'a runmat_value::Value;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(value) = self.pending.pop() {
            if let runmat_value::Value::Cell(cell) = value {
                self.pending.extend(cell.data().iter().rev());
            } else {
                return Some(value);
            }
        }
        None
    }
}
