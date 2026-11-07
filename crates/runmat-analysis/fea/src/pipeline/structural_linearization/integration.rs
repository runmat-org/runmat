//! Physical initial-stress integration. Curved-rule agreement is a convergence
//! check, not a certified error bound; unsupported convergence fails closed.

use super::*;
use crate::assembly::{
    dofs::StructuralDofLayout,
    elements::solid::material::SolidMaterialError,
    elements::solid::response::response_at_volume_point,
    elements::solid::response::SolidResponseError,
    elements::solid::{tetrahedron10_is_affine, Tetrahedron10ElementGeometry},
    SolidRecoveryElementSummary,
};
use crate::topology::{quadrature::*, VolumeQuadraturePoint};
use runmat_meshing_core::ElementOrder;

struct Integrated<const D: usize> {
    matrix: [[f64; D]; D],
    stresses: Vec<[[f64; 3]; 3]>,
    /// Constitutive/gradient magnitude before cancellation, in geometric-stiffness units.
    roundoff_scale: f64,
}

pub(super) fn assemble(
    element: &SolidRecoveryElementSummary,
    layout: &StructuralDofLayout,
    solution: &[f64],
    assembler: &mut RealSparseAssembler,
) -> Result<Vec<[[f64; 3]; 3]>, StructuralLinearizationError> {
    match &element.geometry {
        SolidRecoveryGeometry::Tetrahedron4 { coordinates_m, .. } => assemble_order::<12>(
            element,
            layout,
            solution,
            assembler,
            ElementOrder::Tet4,
            coordinates_m.canonical_values(),
            false,
        ),
        SolidRecoveryGeometry::Tetrahedron10 { coordinates_m, .. } => {
            let geometry = Tetrahedron10ElementGeometry {
                nodes_m: coordinates_m.canonical_value(),
            };
            geometry
                .volume_m3()
                .map_err(|error| element_error(&element.element_id, error))?;
            assemble_order::<30>(
                element,
                layout,
                solution,
                assembler,
                ElementOrder::Tet10,
                coordinates_m.canonical_values(),
                !tetrahedron10_is_affine(geometry),
            )
        }
    }
}

fn assemble_order<const D: usize>(
    element: &SolidRecoveryElementSummary,
    layout: &StructuralDofLayout,
    solution: &[f64],
    assembler: &mut RealSparseAssembler,
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    curved: bool,
) -> Result<Vec<[[f64; 3]; 3]>, StructuralLinearizationError> {
    let mut displacement = Vec::with_capacity(coordinates.len());
    for &node in element.geometry.node_indices() {
        let mut local = [0.0; 3];
        for (axis, kind) in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .enumerate()
        {
            let index = layout
                .index(node, kind)
                .ok_or(StructuralLinearizationError::InvalidSolution)?;
            local[axis] = *solution
                .get(index)
                .ok_or(StructuralLinearizationError::InvalidSolution)?;
        }
        displacement.push(local);
    }
    let integrated = integrate::<D>(element, &displacement, order, coordinates, curved)?;
    assemble_element_matrix(
        assembler,
        layout,
        element.geometry.node_indices(),
        &integrated.matrix,
    )?;
    Ok(integrated.stresses)
}

fn integrate<const D: usize>(
    element: &SolidRecoveryElementSummary,
    displacement: &[[f64; 3]],
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    curved: bool,
) -> Result<Integrated<D>, StructuralLinearizationError> {
    if !curved {
        return integrate_rule(element, displacement, order, coordinates, None);
    }
    let eight = integrate_rule::<D>(
        element,
        displacement,
        order,
        coordinates,
        Some(&GAUSS_LEGENDRE_EIGHT_UNIT),
    )?;
    let twelve = integrate_rule::<D>(
        element,
        displacement,
        order,
        coordinates,
        Some(&GAUSS_LEGENDRE_TWELVE_UNIT),
    )?;
    if converged(&eight, &twelve) {
        return Ok(twelve);
    }
    drop(eight);
    let sixteen = integrate_rule::<D>(
        element,
        displacement,
        order,
        coordinates,
        Some(&GAUSS_LEGENDRE_SIXTEEN_UNIT),
    )?;
    if converged(&twelve, &sixteen) {
        Ok(sixteen)
    } else {
        Err(element_error(
            &element.element_id,
            "curved initial-stress quadrature did not converge",
        ))
    }
}

fn integrate_rule<const D: usize>(
    element: &SolidRecoveryElementSummary,
    displacement: &[[f64; 3]],
    order: ElementOrder,
    coordinates: &[[f64; 3]],
    rule: Option<&[(f64, f64)]>,
) -> Result<Integrated<D>, StructuralLinearizationError> {
    let mut output = Integrated {
        matrix: [[0.0; D]; D],
        stresses: Vec::new(),
        roundoff_scale: 0.0,
    };
    let lambda = element
        .material
        .lame_lambda_pa()
        .map_err(|error| material_error(&element.element_id, error))?;
    let mu = element
        .material
        .shear_modulus_pa()
        .map_err(|error| material_error(&element.element_id, error))?;
    let constitutive_scale = 3.0 * (lambda.abs() + 2.0 * mu.abs());
    let mut failure = None;
    let mut visit = |point: VolumeQuadraturePoint| {
        if failure.is_some() {
            return;
        }
        let response = match response_at_volume_point(element, displacement, &point) {
            Ok(response) => response,
            Err(error) => {
                failure = Some(match error {
                    SolidResponseError::Accounting(error) => {
                        StructuralLinearizationError::Accounting(error)
                    }
                    other => element_error(&element.element_id, other),
                });
                return;
            }
        };
        let s = response.stress;
        let tensor = [[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]];
        let gradients = point.physical_shape_gradients();
        // |a^T sigma b| <= ||a||_1 ||b||_1 max|sigma_ij|. The constitutive
        // row-sum bound below uses total AND elastic strain, so exact free expansion
        // retains a physical scale even when thermal subtraction cancels stress.
        // Absolute quadrature weights also support rules containing negative weights.
        let gradient_scale = gradients
            .iter()
            .map(|gradient| gradient.iter().map(|value| value.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let strain_scale = response
            .total_strain
            .iter()
            .chain(response.elastic_strain.iter())
            .map(|value| value.abs())
            .sum::<f64>();
        output.roundoff_scale += point.volume_weight_m3.abs()
            * constitutive_scale
            * strain_scale
            * gradient_scale
            * gradient_scale;
        for (left, a) in gradients.iter().enumerate() {
            for (right, b) in gradients.iter().enumerate().skip(left) {
                let scalar = (0..3)
                    .map(|i| (0..3).map(|j| a[i] * tensor[i][j] * b[j]).sum::<f64>())
                    .sum::<f64>()
                    * point.volume_weight_m3;
                for axis in 0..3 {
                    output.matrix[left * 3 + axis][right * 3 + axis] += scalar;
                    if left != right {
                        output.matrix[right * 3 + axis][left * 3 + axis] += scalar;
                    }
                }
            }
        }
        output.stresses.push(tensor);
    };
    let valid = if let Some(rule) = rule {
        visit_volume_rule(
            order,
            coordinates,
            tensor_tetrahedron_rule(rule),
            &mut visit,
        )
    } else {
        visit_volume_quadrature(order, coordinates, &mut visit)
    };
    let valid = valid.map_err(StructuralLinearizationError::Accounting)?;
    if let Some(error) = failure {
        return Err(error);
    }
    if valid.is_none()
        || !output.roundoff_scale.is_finite()
        || output
            .matrix
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(element_error(
            &element.element_id,
            "invalid initial-stress quadrature geometry or matrix",
        ));
    }
    Ok(output)
}

fn material_error(element_id: &str, error: SolidMaterialError) -> StructuralLinearizationError {
    match error {
        SolidMaterialError::Accounting(error) => StructuralLinearizationError::Accounting(error),
        other => element_error(element_id, other),
    }
}

fn converged<const D: usize>(previous: &Integrated<D>, next: &Integrated<D>) -> bool {
    let scale = next
        .matrix
        .iter()
        .flatten()
        .chain(previous.matrix.iter().flatten())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    // A modest operation-count allowance for interpolation, constitutive subtraction,
    // quadratic forms and accumulation; not a certified rounding-error estimate.
    // Unlike a fixed absolute floor, this scales with material, strain and length units.
    let physical_scale = previous.roundoff_scale.max(next.roundoff_scale);
    let tolerance = 256.0 * f64::EPSILON * physical_scale + 1.0e-8 * scale;
    scale.is_finite()
        && previous.roundoff_scale.is_finite()
        && next.roundoff_scale.is_finite()
        && previous.roundoff_scale >= 0.0
        && next.roundoff_scale >= 0.0
        && tolerance.is_finite()
        && previous
            .matrix
            .iter()
            .flatten()
            .zip(next.matrix.iter().flatten())
            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod legacy_invariants;
