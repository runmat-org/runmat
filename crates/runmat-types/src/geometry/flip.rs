//! Bounded zero-based axis geometry; out-of-rank axes are singleton no-ops.
#[derive(Clone, Debug)]
pub struct FlipGeometry {
    reversed: Vec<bool>,
    has_effect: bool,
}

impl FlipGeometry {
    pub fn new(shape: &[usize], axes: &[usize]) -> Self {
        let mut reversed = vec![false; shape.len()];
        for &axis in axes {
            if let Some(reversed) = reversed.get_mut(axis) {
                *reversed = !*reversed;
            }
        }
        let has_effect = !shape.contains(&0)
            && reversed
                .iter()
                .zip(shape)
                .any(|(&reverse, &extent)| reverse && extent > 1);
        Self {
            reversed,
            has_effect,
        }
    }
    pub fn reversed_axes(&self) -> &[bool] {
        &self.reversed
    }
    pub fn has_effect(&self) -> bool {
        self.has_effect
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn axes_are_bounded_parity_aware_and_empty_shapes_are_identity() {
        for (shape, axes, effect) in [
            (&[2, 3][..], &[usize::MAX][..], false),
            (&[2, 3], &[0, 0, 1], true),
            (&[1, 3], &[0], false),
            (&[u32::MAX as usize, u32::MAX as usize, 2, 0], &[0], false),
        ] {
            let geometry = FlipGeometry::new(shape, axes);
            assert_eq!(geometry.reversed_axes().len(), shape.len());
            assert_eq!(geometry.has_effect(), effect);
        }
    }
}
