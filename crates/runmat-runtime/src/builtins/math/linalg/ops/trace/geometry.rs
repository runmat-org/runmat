//! Metadata-only selection through trace's actual matrix-shape admission.
use super::*;
pub(in crate::builtins::math) struct TraceGeometry {
    count: usize,
    stride: usize,
}
impl TraceGeometry {
    pub(in crate::builtins::math) fn factor(&self) -> usize {
        self.count
    }
    pub(in crate::builtins::math) fn source(&self, output: usize) -> Option<usize> {
        (output < self.count).then(|| output * self.stride)
    }
    pub(in crate::builtins::math) fn project(&self, input: usize) -> Option<usize> {
        if input.is_multiple_of(self.stride) && input / self.stride < self.count {
            Some(0)
        } else {
            None
        }
    }
}
pub(in crate::builtins::math) fn trace_geometry(shape: &[usize]) -> BuiltinResult<TraceGeometry> {
    let (rows, cols) = ensure_matrix_shape(NAME, shape)?;
    let stride = rows
        .checked_add(1)
        .ok_or_else(|| trace_invalid_input("trace diagonal stride exceeds addressable size"))?;
    Ok(TraceGeometry {
        count: rows.min(cols),
        stride,
    })
}
