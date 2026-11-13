//! Author the same affine-body equalities and let shared assembly lower them.
use super::*;
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, RegionTarget, StructuralConstraint,
    StructuralConstraintEquation, StructuralConstraintEquationTerm, StructuralConstraintKind,
    StructuralTranslationDof,
};
pub(super) fn author_reference_equalities(f: &mut CoupledFixture) {
    let reduction = f.summary.affine_constraint_reduction.as_ref().unwrap();
    let term = |row: usize, coefficient: f64| {
        let (node, axis) = f
            .surface
            .rows
            .iter()
            .enumerate()
            .find_map(|(node, rows)| rows.iter().position(|r| *r == row).map(|axis| (node, axis)))
            .unwrap();
        StructuralConstraintEquationTerm {
            target: RegionTarget::point(format!(
                "node:{}",
                f.surface.projection.nodes[node].node_id
            )),
            dof: [
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ][axis],
            coefficient,
        }
    };
    f.model.constraints = (0..reduction.full_dof_count)
        .filter(|row| !reduction.free_dof_indices.contains(row))
        .map(|row| {
            let mut terms = vec![term(row, 1.)];
            for entry in
                reduction.recovery_row_offsets[row]..reduction.recovery_row_offsets[row + 1]
            {
                let master = reduction.free_dof_indices[reduction.recovery_column_indices[entry]];
                terms.push(term(master, -reduction.recovery_coefficients[entry]));
            }
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("affine_body_{row}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
                    terms,
                    right_hand_side: Displacement::new(0.),
                }),
            })
        })
        .collect();
}
