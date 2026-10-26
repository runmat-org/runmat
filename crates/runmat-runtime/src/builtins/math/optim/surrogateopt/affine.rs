//! Intrinsic coordinates from the original equality nullspace, not a Gram
//! inverse or a cloned linear model. The actual trial point remains unchanged.
use super::{domain::BoxView, linear_view::OriginalRows, IDENTITY, INVALID_ARGUMENT};
use crate::builtins::math::linalg::ops::sparse_factor::orthogonal_frame::OrthogonalSpan;
use crate::builtins::math::optim::common::feasible_cone::Basis;
use crate::builtins::math::optim::common::{
    feasible_cone::{Cone, Diagonal, Normal, Policy},
    ScalarCooperation,
};
use crate::BuiltinResult;
use runmat_value::SparseTensor;

pub(super) use crate::builtins::math::optim::common::feasible_cone::IndependentAxes as Coordinates;
pub(super) async fn coordinates(
    domain: &BoxView<'_>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Coordinates> {
    equality_cone(domain, domain, cooperation)
        .await?
        .independent_axes(cooperation)
        .await
}
async fn equality_cone(
    domain: &BoxView<'_>,
    diagonal: &impl Diagonal,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Cone> {
    let policy = Policy {
        identity: IDENTITY,
        invalid: &INVALID_ARGUMENT,
    };
    let mut cone = Cone::full(policy, domain.dimensions(), cooperation).await?;
    let linear = domain.linear();
    let original_rows = OriginalRows::new(domain, linear.equalities(), cooperation).await?;
    // Only homogeneous original EQUALITY normals constrain this nullspace.
    // Inequality-active tangent cones are a separate sampling object.
    for row in 0..linear.equalities() {
        let coefficients = original_rows.coefficients(row, domain, cooperation).await?;
        let normal = Normal::new(&coefficients, diagonal, policy, cooperation).await?;
        cone.intersect_normal(&normal, false, cooperation).await?;
        cone.intersect_normal(&normal, true, cooperation).await?;
    }
    Ok(cone)
}
pub(super) async fn span_in(
    domain: &BoxView<'_>,
    diagonal: &impl Diagonal,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<OrthogonalSpan>> {
    let cone = equality_cone(domain, diagonal, cooperation).await?;
    let count = cone.len(Basis::Maximal)?;
    if count == 0 || count % 2 != 0 {
        return Ok(None);
    }
    let rank = count / 2;
    // Convert each exact basis column once into the actual sparse QR input.
    // There is no dense n-by-r matrix or retained Q alongside that input.
    let mut pointers = runmat_value::exact_capacity_vec(
        rank.checked_add(1)
            .ok_or_else(|| super::invalid("nullspace pointer extent overflow"))?,
    )
    .map_err(allocation)?;
    let mut indices = Vec::new();
    let mut values = Vec::new();
    pointers.push(0);
    for column in 0..rank {
        let Some(vector) = cone
            .oriented_direction(column, Basis::Maximal, false, cooperation)
            .await?
        else {
            return Ok(None);
        };
        let mut count = 0usize;
        for &value in &vector {
            count += usize::from(value != 0.0);
            cooperation.completed(1).await?;
        }
        indices.try_reserve(count).map_err(reserve)?;
        values.try_reserve(count).map_err(reserve)?;
        for (row, value) in vector.into_iter().enumerate() {
            if value != 0.0 {
                indices.push(row);
                values.push(value);
            }
            cooperation.completed(1).await?;
        }
        pointers.push(indices.len());
    }
    let matrix = {
        let _entry = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        SparseTensor::new(domain.dimensions(), rank, pointers, indices, values)
            .map_err(|detail| super::invalid(format!("invalid nullspace QR topology: {detail}")))?
    };
    let (runtime, budget) = cooperation.runtime_budget();
    let Some(span) = OrthogonalSpan::factor(&matrix, runtime, IDENTITY, budget).await? else {
        return Ok(None);
    };
    Ok(Some(span))
}
fn allocation(source: runmat_value::ExactHostNumericAllocationError) -> crate::RuntimeError {
    crate::build_runtime_error("surrogateopt: nullspace metadata allocation failed")
        .with_builtin(super::NAME)
        .with_source(source)
        .build()
}
fn reserve(source: std::collections::TryReserveError) -> crate::RuntimeError {
    crate::build_runtime_error("surrogateopt: sparse nullspace allocation failed")
        .with_builtin(super::NAME)
        .with_source(source)
        .build()
}
