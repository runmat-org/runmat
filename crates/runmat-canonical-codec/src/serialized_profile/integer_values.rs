//! Exact integer wire widths versus future-value shape bounds.
use super::*;
impl Profiler<'_> {
    pub(super) fn integer(
        self,
        value: i128,
        maximum_width: u64,
        maximum_key_width: usize,
    ) -> Result<(), CanonicalCodecError> {
        if self.key {
            if self.counter.integer_value_bounds {
                // The caller supplies the entire decimal range of its Serde integer type.
                self.counter.string(maximum_key_width, true, self.depth)
            } else {
                self.display(value)
            }
        } else {
            let width = if self.counter.integer_value_bounds {
                maximum_width
            } else {
                wire::integer(value)
            };
            self.scalar(width)
        }
    }
}
