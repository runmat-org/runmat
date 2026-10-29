use super::*;
use crate::{
    algebra::{MatrixProperties, OperatorRole},
    assembly::dofs::StructuralNodeDofSet,
    planning::resource::{execute_scoped_chunks, LocalExecutionPlan, LocalThreadCount},
    progress::{replace_fea_progress_context, FeaCancellationToken, FeaProgressContext},
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
    let mut damping = RealSparseAssembler::square(
        layout.total_dof_count(),
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    assemble_connectors_with_resolver(
        std::slice::from_ref(&connector),
        ConnectorAssemblyContext {
            topology_identity:
                crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
                    mesh_digest: runmat_meshing_core::StableDigest::from_bytes([99; 32]),
                },
            layout: &layout,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            node_identities: &[
                runmat_meshing_core::StableDigest::from_bytes([1; 32]),
                runmat_meshing_core::StableDigest::from_bytes([2; 32]),
            ],
        },
        LocalExecutionPlan::serial(),
        &mut assembler,
        &mut damping,
        |_, target, role| {
            assert_eq!(role, ConnectorTargetRole::Point);
            Ok(ResolvedConnectorTarget {
                node_indices: vec![if target.region_id == "node:1" { 0 } else { 1 }],
                surface_samples: Vec::new(),
            })
        },
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

#[test]
fn scoped_surface_foundation_preserves_canonical_sparse_accumulation_across_waves() {
    let layout =
        StructuralDofLayout::from_node_sets(vec![StructuralNodeDofSet::translational(); 6]);
    let connector = AnalysisConnector {
        connector_id: "foundation".into(),
        kind: ConnectorKind::SurfaceNormalFoundation {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:1")),
            modulus: runmat_analysis_core::FoundationModulus::new(20.0),
        },
    };
    let samples = (0..29)
        .map(|ordinal| SurfaceFoundationSample {
            node_indices: (0..6).collect(),
            shape_values: vec![
                0.1 + ordinal as f64 * 0.001,
                0.2,
                0.15,
                0.2,
                0.15,
                0.2 - ordinal as f64 * 0.001,
            ],
            unit_normal: [0.25, 0.5, 0.75],
            area_weight_m2: 0.01 + ordinal as f64 * 0.0001,
        })
        .collect::<Vec<_>>();
    let assemble = || {
        RealSparseAssembler::square(
            layout.total_dof_count(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
    };
    let mut serial = assemble();
    add_surface_normal_foundation(&mut serial, &layout, &samples, 20.0);
    let reference = serial.finish().unwrap();
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    for _ in 0..2 {
        let mut scoped = assemble();
        scoped::add_surface_normal_foundation_scoped(
            &connector,
            &mut scoped,
            &layout,
            &samples,
            20.0,
            plan,
        )
        .unwrap();
        assert_eq!(scoped.finish().unwrap(), reference);
    }
    assert!(
        std::mem::size_of::<(usize, usize, f64)>() * 18 * 18 * 4 <= 128 * 1024,
        "a worker's retained connector wave must fit the fixed assembly scratch"
    );
    let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
        let mut scoped = assemble();
        Ok::<_, ()>(scoped::add_surface_normal_foundation_scoped(
            &connector,
            &mut scoped,
            &layout,
            &samples,
            20.0,
            plan,
        ))
    })
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(StructuralAssemblyError::LocalExecution(message))
            if message.contains("nested local numerical execution")
    ));

    let cancellation = FeaCancellationToken::new();
    cancellation.cancel();
    let _context = replace_fea_progress_context(FeaProgressContext {
        cancellation,
        ..FeaProgressContext::default()
    });
    let mut cancelled = assemble();
    assert!(matches!(
        scoped::add_surface_normal_foundation_scoped(
            &connector,
            &mut cancelled,
            &layout,
            &samples,
            20.0,
            plan,
        ),
        Err(StructuralAssemblyError::Cancelled)
    ));
    drop(_context);

    let mut oversized = samples.clone();
    oversized[0].node_indices.push(0);
    oversized[0].shape_values.push(0.0);
    assert!(matches!(
        scoped::validate_surface_samples(&connector, &oversized),
        Err(StructuralAssemblyError::InvalidExplicitConnector { connector_id, .. })
            if connector_id == "foundation"
    ));
}

#[test]
fn scoped_connector_assembly_preserves_plan_provenance_and_mutable_resolution() {
    let layout =
        StructuralDofLayout::from_node_sets(vec![StructuralNodeDofSet::translational(); 6]);
    let connector = AnalysisConnector {
        connector_id: "foundation".into(),
        kind: ConnectorKind::SurfaceNormalFoundation {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:1")),
            modulus: runmat_analysis_core::FoundationModulus::new(20.0),
        },
    };
    let coordinates = (0..6)
        .map(|index| [index as f64, 0.0, 0.0])
        .collect::<Vec<_>>();
    let identities = (1..=6)
        .map(|value| runmat_meshing_core::StableDigest::from_bytes([value; 32]))
        .collect::<Vec<_>>();
    let samples = (0..29)
        .map(|ordinal| SurfaceFoundationSample {
            node_indices: (0..6).collect(),
            shape_values: vec![
                0.1 + ordinal as f64 * 0.001,
                0.2,
                0.15,
                0.2,
                0.15,
                0.2 - ordinal as f64 * 0.001,
            ],
            unit_normal: [0.0, 0.0, 1.0],
            area_weight_m2: 0.01,
        })
        .collect::<Vec<_>>();
    let assemble = |execution| {
        let mut stiffness = RealSparseAssembler::square(
            layout.total_dof_count(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        let mut damping = RealSparseAssembler::square(
            layout.total_dof_count(),
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        let mut resolution_count = 0;
        let summary = assemble_connectors_with_resolver(
            std::slice::from_ref(&connector),
            ConnectorAssemblyContext {
                topology_identity:
                    crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
                        mesh_digest: runmat_meshing_core::StableDigest::from_bytes([99; 32]),
                    },
                layout: &layout,
                coordinates_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Position,_>::new(&coordinates),
                node_identities: &identities,
            },
            execution,
            &mut stiffness,
            &mut damping,
            |_, _, role| {
                assert_eq!(role, ConnectorTargetRole::Surface);
                resolution_count += 1;
                Ok(ResolvedConnectorTarget {
                    node_indices: (0..6).collect(),
                    surface_samples: samples.clone(),
                })
            },
        )
        .unwrap();
        assert_eq!(resolution_count, 1);
        (
            summary,
            stiffness.finish().unwrap(),
            damping.finish().unwrap(),
        )
    };
    let reference = assemble(LocalExecutionPlan::serial());
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    assert_eq!(assemble(plan), reference);
    assert_eq!(assemble(plan), reference);
    assert_eq!(reference.0.stiffness_count, 1);
    assert_eq!(reference.0.damping_count, 0);
    assert_eq!(reference.1.contributions().len(), 1);
}

#[test]
fn mixed_point_connector_waves_match_serial_authored_order_and_fail_closed() {
    let layout =
        StructuralDofLayout::from_node_sets(vec![StructuralNodeDofSet::translational(); 6]);
    let coordinates = (0..6)
        .map(|index| [index as f64, index as f64 * 0.25, index as f64 * 0.5])
        .collect::<Vec<_>>();
    let identities = (1..=6)
        .map(|value| runmat_meshing_core::StableDigest::from_bytes([value; 32]))
        .collect::<Vec<_>>();
    let mut connectors = (0..29)
        .map(|index| {
            let primary = RegionTarget::point(format!("node:{}", index % 6));
            let secondary = RegionTarget::point(format!("node:{}", (index + 1) % 6));
            AnalysisConnector {
                connector_id: format!("point_{index}"),
                kind: match index % 3 {
                    0 => ConnectorKind::AxialSpring {
                        primary,
                        secondary,
                        stiffness: runmat_analysis_core::Stiffness::new(10.0 + index as f64),
                    },
                    1 => ConnectorKind::AxialDashpot {
                        primary,
                        secondary,
                        damping: runmat_analysis_core::Damping::new(5.0 + index as f64),
                    },
                    _ => ConnectorKind::IsotropicGroundSpring {
                        target: primary,
                        stiffness: runmat_analysis_core::Stiffness::new(20.0 + index as f64),
                    },
                },
            }
        })
        .collect::<Vec<_>>();
    connectors.insert(
        11,
        AnalysisConnector {
            connector_id: "interleaved_foundation".into(),
            kind: ConnectorKind::SurfaceNormalFoundation {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:1",
                )),
                modulus: runmat_analysis_core::FoundationModulus::new(12.0),
            },
        },
    );
    let assemble = |execution| {
        let mut stiffness = RealSparseAssembler::square(
            layout.total_dof_count(),
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        let mut damping = RealSparseAssembler::square(
            layout.total_dof_count(),
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        );
        let mut resolution_count = 0;
        let summary = assemble_connectors_with_resolver(
            &connectors,
            ConnectorAssemblyContext {
                topology_identity: crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
                    mesh_digest: runmat_meshing_core::StableDigest::from_bytes([99; 32]),
                },
                layout: &layout,
                coordinates_m: runmat_types::physical::Quantity::<runmat_types::physical::kind::Position,_>::new(&coordinates),
                node_identities: &identities,
            },
            execution,
            &mut stiffness,
            &mut damping,
            |_, target, role| {
                resolution_count += 1;
                match role {
                    ConnectorTargetRole::Point => Ok(ResolvedConnectorTarget {
                        node_indices: vec![target.region_id.strip_prefix("node:").unwrap().parse::<usize>().unwrap()],
                        surface_samples: Vec::new(),
                    }),
                    ConnectorTargetRole::Surface => Ok(ResolvedConnectorTarget {
                        node_indices: vec![0, 1, 2],
                        surface_samples: vec![SurfaceFoundationSample {
                            node_indices: vec![0, 1, 2],
                            shape_values: vec![0.2, 0.3, 0.5],
                            unit_normal: [0.0, 0.0, 1.0],
                            area_weight_m2: 0.5,
                        }],
                    }),
                }
            },
        )?;
        Ok::<_, StructuralAssemblyError>((
            summary,
            stiffness.finish().unwrap(),
            damping.finish().unwrap(),
            resolution_count,
        ))
    };
    let reference = assemble(LocalExecutionPlan::serial()).unwrap();
    assert_eq!(reference.0.stiffness_count, 20);
    assert_eq!(reference.0.damping_count, 10);
    assert_eq!(reference.3, 50);
    for threads in [2, 3, 7] {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(threads).unwrap()).unwrap();
        for _ in 0..2 {
            assert_eq!(assemble(plan).unwrap(), reference);
        }
    }
    let worker_wave_bytes = std::mem::size_of::<PreparedPointConnector<'_>>() * 8
        + std::mem::size_of::<Vec<(usize, usize, f64)>>() * 8 * 2
        + std::mem::size_of::<(usize, usize, f64)>() * 36 * 8 * 2;
    assert!(worker_wave_bytes < 128 * 1024);

    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
        Ok::<_, ()>(assemble(plan))
    })
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(StructuralAssemblyError::LocalExecution(message))
            if message.contains("nested local numerical execution")
    ));
    let cancellation = FeaCancellationToken::new();
    cancellation.cancel();
    let _context = replace_fea_progress_context(FeaProgressContext {
        cancellation,
        ..FeaProgressContext::default()
    });
    assert!(matches!(
        assemble(plan),
        Err(StructuralAssemblyError::Cancelled)
    ));
}

#[test]
fn scoped_point_connector_keeps_authored_error_attribution() {
    let layout = StructuralDofLayout::from_node_sets(vec![StructuralNodeDofSet::from_kinds([
        StructuralDofKind::Rx,
    ])]);
    let connector = AnalysisConnector {
        connector_id: "rotation_only_ground".into(),
        kind: ConnectorKind::IsotropicGroundSpring {
            target: RegionTarget::point("node:0"),
            stiffness: runmat_analysis_core::Stiffness::new(10.0),
        },
    };
    let mut stiffness = RealSparseAssembler::square(
        layout.total_dof_count(),
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let mut damping = RealSparseAssembler::square(
        layout.total_dof_count(),
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    );
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
    let error = assemble_connectors_with_resolver(
        &[connector],
        ConnectorAssemblyContext {
            topology_identity:
                crate::physics::structural::topology::CanonicalStructuralTopologyIdentity::Solid {
                    mesh_digest: runmat_meshing_core::StableDigest::from_bytes([99; 32]),
                },
            layout: &layout,
            coordinates_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(&[[0.0, 0.0, 0.0]]),
            node_identities: &[runmat_meshing_core::StableDigest::from_bytes([1; 32])],
        },
        plan,
        &mut stiffness,
        &mut damping,
        |_, _, role| {
            assert_eq!(role, ConnectorTargetRole::Point);
            Ok(ResolvedConnectorTarget {
                node_indices: vec![0],
                surface_samples: Vec::new(),
            })
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, StructuralAssemblyError::InvalidExplicitConnector {
        connector_id, reason: "ground spring node lacks a translational degree of freedom"
    } if connector_id == "rotation_only_ground")
    );
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
