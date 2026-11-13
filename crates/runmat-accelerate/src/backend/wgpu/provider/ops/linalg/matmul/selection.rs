//! One physical WGPU matmul pipeline selector shared by execution and cache inspection.

use crate::backend::wgpu::types::NumericPrecision;

use super::K_CHUNK_SWITCH;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::backend::wgpu::provider) enum MatmulPipelineRoute {
    Empty,
    Scalar,
    Vec4,
    Chunked,
}

pub(in crate::backend::wgpu::provider) fn selected_matmul_pipeline(
    precision: NumericPrecision,
    m: usize,
    n: usize,
    k: usize,
    transpose_a: bool,
    transpose_b: bool,
) -> MatmulPipelineRoute {
    if m == 0 || n == 0 {
        return MatmulPipelineRoute::Empty;
    }
    if !transpose_a && !transpose_b && k >= K_CHUNK_SWITCH {
        return MatmulPipelineRoute::Chunked;
    }
    let vec4_disabled = std::env::var("RUNMAT_DISABLE_VEC4")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "True"))
        .unwrap_or(false);
    if precision == NumericPrecision::F32
        && !transpose_a
        && !transpose_b
        && m >= 4
        && m.is_multiple_of(4)
        && k < K_CHUNK_SWITCH
        && !vec4_disabled
    {
        MatmulPipelineRoute::Vec4
    } else {
        MatmulPipelineRoute::Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_respects_geometry_and_operand_views() {
        assert_eq!(
            selected_matmul_pipeline(NumericPrecision::F64, 0, 2, 3, false, false),
            MatmulPipelineRoute::Empty
        );
        assert_eq!(
            selected_matmul_pipeline(NumericPrecision::F64, 4, 2, 3, false, false),
            MatmulPipelineRoute::Scalar
        );
        assert_eq!(
            selected_matmul_pipeline(NumericPrecision::F32, 4, 2, K_CHUNK_SWITCH, false, false),
            MatmulPipelineRoute::Chunked
        );
        assert_eq!(
            selected_matmul_pipeline(NumericPrecision::F32, 4, 2, 3, true, false),
            MatmulPipelineRoute::Scalar
        );
    }
}
