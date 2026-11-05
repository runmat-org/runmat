//! Isoparametric Tet10 geometry, gradients, and whole-element Jacobian admission.

use runmat_simulation_usage::{OperationCount, UsageCategory, UsageContext};

use super::tetrahedron10::{
    Tetrahedron10CapacityError, Tetrahedron10ElementError, Tetrahedron10ElementGeometry,
    TETRAHEDRON10_ELEMENT_NODE_COUNT,
};

pub(super) fn physical_shape_gradients_and_measure(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    barycentric: [f64; 4],
) -> Result<([[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT], f64), Tetrahedron10ElementError> {
    let reference = admitted_reference_shape_gradients(barycentric)?;
    let (inverse, determinant) = inverse_jacobian(nodes_m, reference)?;
    // Ten shape functions, three axes, each a three-term dot (3 products,
    // 2 additions). The inverse Jacobian has its own reservation.
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(150),
            )
        })
        .transpose()?;
    let gradients = reference.map(|gradient| {
        std::array::from_fn(|axis| {
            inverse[axis][0] * gradient[0]
                + inverse[axis][1] * gradient[1]
                + inverse[axis][2] * gradient[2]
        })
    });
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(150))?;
    }
    Ok((gradients, determinant))
}

pub(super) fn reference_shape_gradients(
    barycentric: [f64; 4],
) -> [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT] {
    runmat_meshing_core::tetrahedron10_reference_gradients(barycentric)
}

fn admitted_reference_shape_gradients(
    barycentric: [f64; 4],
) -> Result<[[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT], Tetrahedron10ElementError> {
    // Four corner factors and their three scales (20), plus six edge rows
    // with three axes of two products, one add, and one final scale (72).
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, OperationCount::from_operations(92))
        })
        .transpose()?;
    let gradients = reference_shape_gradients(barycentric);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(92))?;
    }
    Ok(gradients)
}

pub(super) fn jacobian_determinant(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    barycentric: [f64; 4],
) -> Result<f64, Tetrahedron10ElementError> {
    let reference = admitted_reference_shape_gradients(barycentric)?;
    // Affine Jacobian: nine differences; curved: nine ten-term dots.
    // Determinant is a three-vector cross (9) and dot (5).
    let count = if is_affine(Tetrahedron10ElementGeometry { nodes_m }) {
        23
    } else {
        185
    };
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(count),
            )
        })
        .transpose()?;
    let jacobian = jacobian(nodes_m, reference);
    let determinant = dot(jacobian[0], cross(jacobian[1], jacobian[2]));
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(count))?;
    }
    if determinant.is_finite() && determinant > 0.0 {
        Ok(determinant)
    } else {
        Err(Tetrahedron10ElementError::DegenerateOrInverted)
    }
}

fn inverse_jacobian(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    reference: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> Result<([[f64; 3]; 3], f64), Tetrahedron10ElementError> {
    let affine = is_affine(Tetrahedron10ElementGeometry { nodes_m });
    let prefix = if affine { 23 } else { 185 };
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(prefix + 37),
            )
        })
        .transpose()?;
    let matrix = jacobian(nodes_m, reference);
    let determinant = dot(matrix[0], cross(matrix[1], matrix[2]));
    if !determinant.is_finite() || determinant <= 0.0 {
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(prefix))?;
        }
        return Err(Tetrahedron10ElementError::DegenerateOrInverted);
    }
    let inverse_determinant = 1.0 / determinant;
    let result = (
        [
            [
                (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inverse_determinant,
                (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inverse_determinant,
                (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inverse_determinant,
            ],
            [
                (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inverse_determinant,
                (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inverse_determinant,
                (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inverse_determinant,
            ],
            [
                (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inverse_determinant,
                (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inverse_determinant,
                (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inverse_determinant,
            ],
        ],
        determinant,
    );
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(prefix + 37))?;
    }
    Ok(result)
}

fn jacobian(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    reference: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> [[f64; 3]; 3] {
    // Exactly affine Tet10 nodes have a constant Jacobian. Deriving it from
    // corner differences avoids cancellation of quadratic shape derivatives,
    // and does not approximate or discard any curved geometry.
    if is_affine(Tetrahedron10ElementGeometry { nodes_m }) {
        return std::array::from_fn(|reference_axis| {
            std::array::from_fn(|physical_axis| {
                nodes_m[reference_axis + 1][physical_axis] - nodes_m[0][physical_axis]
            })
        });
    }
    std::array::from_fn(|reference_axis| {
        std::array::from_fn(|physical_axis| {
            (0..TETRAHEDRON10_ELEMENT_NODE_COUNT)
                .map(|node| nodes_m[node][physical_axis] * reference[node][reference_axis])
                .sum()
        })
    })
}

pub(crate) fn is_affine(geometry: Tetrahedron10ElementGeometry) -> bool {
    runmat_meshing_core::tetrahedron10_is_affine(&geometry.nodes_m)
}

/// Admits the element by the shared Tet10 validity rule's Jacobian margin
/// (`runmat_meshing_core::tetrahedron10_jacobian_margin`), the same margin the mesher certifies.
pub(super) fn validate_nodes(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
) -> Result<(), Tetrahedron10ElementError> {
    if !nodes_m.iter().flatten().all(|value| value.is_finite()) {
        return Err(Tetrahedron10ElementError::NonFiniteCoordinate);
    }
    match runmat_meshing_core::tetrahedron10_jacobian_margin_for_category(
        &nodes_m,
        UsageCategory::Assembly,
    ) {
        Err(error) => Err(Tetrahedron10ElementError::Accounting(error)),
        Ok(None) => Err(Tetrahedron10ElementError::DegenerateOrInverted),
        Ok(Some(margin)) if margin.is_valid() => Ok(()),
        Ok(Some(margin)) if margin.minimum_jacobian_lower_bound <= 0.0 => {
            Err(Tetrahedron10ElementError::DegenerateOrInverted)
        }
        Ok(Some(_)) => Err(Tetrahedron10ElementError::UncertifiedJacobian),
    }
}

pub(super) async fn validate_nodes_with_capacity<E>(
    nodes_m: [[f64; 3]; TETRAHEDRON10_ELEMENT_NODE_COUNT],
    owner: &dyn runmat_simulation_usage::AsyncCapacityOwner<E>,
) -> Result<(), Tetrahedron10CapacityError<E>> {
    if !nodes_m.iter().flatten().all(|value| value.is_finite()) {
        return Err(Tetrahedron10ElementError::NonFiniteCoordinate.into());
    }
    let margin = runmat_meshing_core::tetrahedron10_jacobian_margin_with_capacity(
        &nodes_m,
        UsageCategory::Assembly,
        Some(owner),
    )
    .await
    .map_err(|error| match error {
        runmat_meshing_core::Tet10CapacityError::Invalidity(
            runmat_meshing_core::Tet10ElementInvalidity::Accounting(error),
        ) => Tetrahedron10CapacityError::Numerical(Tetrahedron10ElementError::Accounting(error)),
        runmat_meshing_core::Tet10CapacityError::Invalidity(_) => {
            Tetrahedron10CapacityError::Numerical(Tetrahedron10ElementError::DegenerateOrInverted)
        }
        runmat_meshing_core::Tet10CapacityError::Admission(error) => {
            Tetrahedron10CapacityError::Admission(error)
        }
    })?;
    match margin {
        None => Err(Tetrahedron10ElementError::DegenerateOrInverted.into()),
        Some(margin) if margin.is_valid() => Ok(()),
        Some(margin) if margin.minimum_jacobian_lower_bound <= 0.0 => {
            Err(Tetrahedron10ElementError::DegenerateOrInverted.into())
        }
        Some(_) => Err(Tetrahedron10ElementError::UncertifiedJacobian.into()),
    }
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
mod tests;
