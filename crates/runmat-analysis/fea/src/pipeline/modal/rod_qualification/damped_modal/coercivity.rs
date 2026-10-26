//! Diagnostic only: replay shared physical element assembly against the actual
//! original constrained operator. Production must capture these blocks once.
use super::*;
use crate::{
    algebra::{
        anchored_energy::{AnchoredEnergyCheck, EnergyVerdict},
        CoercivityWorkEnvelope,
    },
    assembly::{
        elements::solid::{
            global_stiffness_matrix, tetrahedron10_global_stiffness_matrix,
            Tetrahedron10ElementGeometry, Tetrahedron4ElementGeometry,
        },
        SolidRecoveryGeometry,
    },
};

#[test]
fn original_axial_constraint_spaces_certify_with_actual_reduced_operator_caps() {
    for (quadratic, refinements) in [(false, vec![2, 4, 8, 16]), (true, vec![1, 2, 4])] {
        for cells in refinements {
            let mesh = rod_mesh(cells, quadratic);
            let mut model = rod_model(&mesh, &[1.0]);
            damped::add_damping(&mut model);
            let assembly = assemble(&model, &mesh, 1.0);
            let prepared = match &assembly.real.constraint_matrix {
                Some(constraints) => assembly
                    .real
                    .operator
                    .prepare_modal_with_reduction(constraints.affine_reduction().unwrap())
                    .unwrap(),
                None => assembly.real.operator.prepare_modal().unwrap(),
            };
            let matrix = prepared.operators().stiffness();
            let mut selection = vec![None; mesh.topology.nodes.len()];
            for (row, &full) in prepared.free_dofs().iter().enumerate() {
                assert_eq!(
                    full % 3,
                    0,
                    "original space contains only axial translations"
                );
                selection[full / 3] = Some(row);
            }
            let reduction = prepared.reduction();
            for full in 0..reduction.full_dof_count {
                assert_eq!(reduction.offset[full], 0.0);
                let range =
                    reduction.recovery_row_offsets[full]..reduction.recovery_row_offsets[full + 1];
                match (full % 3 == 0).then_some(selection[full / 3]).flatten() {
                    Some(row) => {
                        assert_eq!(&reduction.recovery_column_indices[range.clone()], &[row]);
                        assert_eq!(&reduction.recovery_coefficients[range], &[1.0]);
                    }
                    None => assert!(range.is_empty()),
                }
            }
            let envelope = CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap();
            let mut check =
                AnchoredEnergyCheck::new_selection(matrix, &selection, envelope).unwrap();
            for element in &assembly.real.structural_solid_recovery {
                let values = match &element.geometry {
                    SolidRecoveryGeometry::Tetrahedron4 { coordinates_m, .. } => {
                        let k = global_stiffness_matrix(
                            element.material,
                            Tetrahedron4ElementGeometry {
                                nodes_m: coordinates_m.canonical_value(),
                            },
                        )
                        .unwrap();
                        (0..4)
                            .flat_map(|i| (0..4).map(move |j| k[3 * i][3 * j]))
                            .collect::<Vec<_>>()
                    }
                    SolidRecoveryGeometry::Tetrahedron10 { coordinates_m, .. } => {
                        let k = tetrahedron10_global_stiffness_matrix(
                            element.material,
                            Tetrahedron10ElementGeometry {
                                nodes_m: coordinates_m.canonical_value(),
                            },
                        )
                        .unwrap();
                        (0..10)
                            .flat_map(|i| (0..10).map(move |j| k[3 * i][3 * j]))
                            .collect::<Vec<_>>()
                    }
                };
                check
                    .push_block(element.geometry.node_indices(), &values)
                    .unwrap();
            }
            let EnergyVerdict::Verified(proof) = check.finish().unwrap() else {
                panic!(
                    "original cells{cells}/Tet{} operator must certify",
                    if quadratic { 10 } else { 4 }
                );
            };
            assert_eq!(
                proof.operator_identity(),
                matrix.numerical_identity().unwrap()
            );
            assert!(proof.lower_bound() > 0.0);
            eprintln!("axial coercivity cells{cells} Tet{} dofs{} lower={} error={} operations={}/{} bytes={}/{}",
                if quadratic {10} else {4},matrix.rows(), proof.lower_bound(), proof.assembly_error_upper(),
                proof.scalar_operations(), envelope.update_limit(),proof.resident_bytes(), envelope.resident_bytes_limit());
        }
    }
}
