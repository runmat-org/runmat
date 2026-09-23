use super::*;
use crate::{
    algebra::{MatrixProperties, OperatorRole},
    assembly::dofs::StructuralNodeDofSet,
};

#[test]
fn axial_operator_is_objective_under_infinitesimal_rigid_rotation() {
    let layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational(),
        StructuralNodeDofSet::translational(),
    ]);
    let connector = AnalysisConnector {
        connector_id: "axial".into(),
        kind: ConnectorKind::AxialSpring {
            primary: RegionTarget::point("node:1"),
            secondary: RegionTarget::point("node:2"),
            stiffness: runmat_analysis_core::Stiffness::new(100.0),
        },
    };
    let mut assembler = RealSparseAssembler::square(
        layout.total_dof_count(),
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    add_axial_operator(
        &mut assembler,
        &layout,
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [0, 1],
        100.0,
        &connector,
    )
    .unwrap();
    let matrix = assembler.finish().unwrap();
    let rigid_rotation = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    assert_eq!(quadratic_form(&matrix, &rigid_rotation), 0.0);
    let axial_extension = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0];
    assert!((quadratic_form(&matrix, &axial_extension) - 6.25).abs() < 1.0e-12);
}

#[test]
fn surface_foundation_scales_with_area_not_node_count() {
    let coarse = foundation_rigid_normal_stiffness(
        1,
        vec![SurfaceFoundationSample {
            node_indices: vec![0],
            shape_values: vec![1.0],
            unit_normal: [0.0, 0.0, 1.0],
            area_weight_m2: 2.0,
        }],
    );
    let refined = foundation_rigid_normal_stiffness(
        2,
        vec![
            SurfaceFoundationSample {
                node_indices: vec![0],
                shape_values: vec![1.0],
                unit_normal: [0.0, 0.0, 1.0],
                area_weight_m2: 1.0,
            },
            SurfaceFoundationSample {
                node_indices: vec![1],
                shape_values: vec![1.0],
                unit_normal: [0.0, 0.0, 1.0],
                area_weight_m2: 1.0,
            },
        ],
    );
    assert!((coarse - 20.0).abs() < 1.0e-12);
    assert!((refined - coarse).abs() < 1.0e-12);
}

fn foundation_rigid_normal_stiffness(
    node_count: usize,
    samples: Vec<SurfaceFoundationSample>,
) -> f64 {
    let layout = StructuralDofLayout::from_node_sets(vec![
        StructuralNodeDofSet::translational();
        node_count
    ]);
    let mut assembler = RealSparseAssembler::square(
        layout.total_dof_count(),
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    add_surface_normal_foundation(&mut assembler, &layout, &samples, 10.0);
    let matrix = assembler.finish().unwrap();
    let mut rigid_normal = vec![0.0; layout.total_dof_count()];
    for node in 0..node_count {
        rigid_normal[layout.index(node, StructuralDofKind::Uz).unwrap()] = 1.0;
    }
    quadratic_form(&matrix, &rigid_normal)
}

fn quadratic_form(matrix: &crate::algebra::RealSparseMatrix, vector: &[f64]) -> f64 {
    (0..vector.len())
        .map(|row| {
            (0..vector.len())
                .map(|column| vector[row] * matrix.get(row, column) * vector[column])
                .sum::<f64>()
        })
        .sum()
}
