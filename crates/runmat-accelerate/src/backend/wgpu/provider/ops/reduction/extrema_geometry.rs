use crate::backend::wgpu::types::NumericPrecision;
use runmat_accelerate_api::unsupported_provider_operation;

pub(super) struct Geometry {
    pub rows: u32,
    pub cols: u32,
    pub length: usize,
    pub shape: Vec<usize>,
    pub groups: u32,
}

/// The shader has 32-bit addressing, one invocation per output, and native
/// floating indices. Reject unrepresentable requests before any materialization.
pub(super) fn geometry(
    shape: &[usize],
    dim: usize,
    precision: NumericPrecision,
    workgroup: u32,
    max_groups: u32,
) -> anyhow::Result<Geometry> {
    let unsupported = || {
        unsupported_provider_operation(
            "dimensional extrema geometry or exact index exceeds this kernel's capability",
        )
    };
    if shape.len() != 2 || dim > 1 || workgroup == 0 {
        return Err(unsupported());
    }
    if shape.contains(&0) {
        return Ok(Geometry {
            rows: 0,
            cols: 0,
            length: 0,
            shape: shape.to_vec(),
            groups: 0,
        });
    }
    let total = shape[0].checked_mul(shape[1]).ok_or_else(unsupported)?;
    if total > u32::MAX as usize || (precision == NumericPrecision::F32 && shape[dim] > (1 << 24)) {
        return Err(unsupported());
    }
    let length = shape[1 - dim];
    let groups = (length as u64).div_ceil(u64::from(workgroup));
    if groups > u64::from(max_groups) {
        return Err(unsupported());
    }
    Ok(Geometry {
        rows: shape[0] as u32,
        cols: shape[1] as u32,
        length,
        shape: if dim == 0 {
            vec![1, shape[1]]
        } else {
            vec![shape[0], 1]
        },
        groups: groups as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_match_frozen_workgroup_and_native_index_precision() {
        for workgroup in [64, 256, 512] {
            let capacity = workgroup as usize * 3;
            assert_eq!(
                geometry(&[2, capacity], 0, NumericPrecision::F32, workgroup, 3)
                    .unwrap()
                    .groups,
                3
            );
            assert!(geometry(&[2, capacity + 1], 0, NumericPrecision::F32, workgroup, 3).is_err());
        }
        assert!(geometry(&[1 << 24, 1], 0, NumericPrecision::F32, 64, 65535).is_ok());
        assert!(geometry(&[(1 << 24) + 1, 1], 0, NumericPrecision::F32, 64, 65535).is_err());
        assert!(geometry(&[(1 << 24) + 1, 1], 0, NumericPrecision::F64, 64, 65535).is_ok());
        for shape in [vec![0, 3], vec![3, 0], vec![usize::MAX, 0]] {
            let plan = geometry(&shape, 1, NumericPrecision::F32, 64, 1).unwrap();
            assert_eq!(plan.shape, shape);
            assert_eq!((plan.length, plan.groups), (0, 0));
        }
        assert!(geometry(&[2, 3, 4], 0, NumericPrecision::F32, 64, 65535).is_err());
    }
}
