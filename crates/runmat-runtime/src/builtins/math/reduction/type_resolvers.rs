//! Legacy linalg shape projection still used by matrix type adapters.
//! Numerical reduction inference is owned by the canonical catalog leaves.
use runmat_builtins::shape_rules::scalar_tensor_shape;

pub fn reduce_first_nonsingleton(shape: &[Option<usize>]) -> Vec<Option<usize>> {
    if shape.is_empty() {
        return scalar_tensor_shape();
    }
    let mut dim = 0usize;
    for (i, entry) in shape.iter().enumerate() {
        if let Some(value) = entry {
            if *value != 1 {
                dim = i;
                break;
            }
        } else {
            dim = 0;
            break;
        }
    }
    let mut out = shape.to_vec();
    if let Some(entry) = out.get_mut(dim) {
        *entry = Some(1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linalg_projection_preserves_the_existing_first_axis_contract() {
        assert_eq!(
            reduce_first_nonsingleton(&[Some(3), Some(4)]),
            vec![Some(1), Some(4)]
        );
        assert_eq!(
            reduce_first_nonsingleton(&[Some(1), Some(4), Some(2)]),
            vec![Some(1), Some(1), Some(2)]
        );
        assert_eq!(
            reduce_first_nonsingleton(&[None, Some(4)]),
            vec![Some(1), Some(4)]
        );
        assert_eq!(reduce_first_nonsingleton(&[]), scalar_tensor_shape());
    }
}
