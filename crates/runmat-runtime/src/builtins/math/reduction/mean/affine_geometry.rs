//! Metadata-only projection through mean's actual argument/weight grammar.
//! No numeric input/probe is fabricated, and no coefficient algebra lives here.
use super::super::sum::{reduction_geometry_for_axes, ReductionGeometry};
use super::*;
pub(in crate::builtins::math) struct AffineGeometry {
    pub reduction: ReductionGeometry,
    weights: Option<(MeanWeights, WeightLayout, usize, usize)>,
}
impl AffineGeometry {
    pub(in crate::builtins::math) fn is_weighted(&self) -> bool {
        self.weights.is_some()
    }
    pub(in crate::builtins::math) fn weight(&self, row: usize) -> f64 {
        match &self.weights {
            None => 1.,
            Some((weights, WeightLayout::Full, _, _)) => weights.f64_at(row),
            Some((weights, WeightLayout::Vector, stride, extent)) => {
                weights.f64_at((row / *stride) % *extent)
            }
        }
    }
}
pub(in crate::builtins::math) async fn affine_geometry(
    shape: &[usize],
    arguments: crate::call::FunctionArgumentView<'_>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: Option<&mut crate::execution::CpuCooperationBudget>,
) -> BuiltinResult<AffineGeometry> {
    crate::context::host_allocation::require_unbounded_workspace(
        runtime,
        "mean affine geometry",
        "complete metadata planning workspace is not certified",
    )?;
    crate::context::RuntimeScopeFactory::new(runtime.cloned(), move || async move {
        let parsed = parse_arguments_cooperative(arguments, cooperation, runtime).await?;
        let axes = resolve_native_dims(shape, &parsed.axes)?;
        let reduction = reduction_geometry_for_axes(shape, &axes)?;
        let weights = match parsed.weights {
            None => None,
            Some(weights) => {
                let dimension = weighted_dimension(shape, &parsed.axes)?;
                let axis = dimension - 1;
                let extent = shape.get(axis).copied().unwrap_or(1);
                let layout = validate_weight_shape(&weights, shape, extent)?;
                let stride = shape
                    .iter()
                    .take(axis)
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .ok_or_else(|| {
                        mean_invalid_input("mean affine stride exceeds addressable size")
                    })?;
                Some((weights, layout, stride, extent))
            }
        };
        Ok(AffineGeometry { reduction, weights })
    })
    .construct()
    .await
}
