use super::*;
use crate::constraints::compilation::topology::CompilationTopology;
use runmat_analysis_core::{
    AnalysisConstraint, Position, ReferencePoint, ReferencePointCoupling, RegionTarget,
};
use runmat_meshing_core::ElementOrder;

fn check(model: &runmat_analysis_core::AnalysisModel, mesh: Option<&SolverMeshArtifact>) {
    let topology = CompilationTopology::derive(model, mesh).unwrap();
    for authored in &model.constraints {
        let constraint = authored.structural().unwrap();
        let mixed = match &topology {
            CompilationTopology::Solid(_) => None,
            CompilationTopology::Explicit { structural, nodes }
            | CompilationTopology::Mixed {
                structural, nodes, ..
            } => Some((*structural, nodes)),
        };
        let prepared = PreparedTargets::derive(constraint, mesh, mixed).unwrap();
        let original: Vec<_> = prepared
            .selections()
            .map(|(target, nodes)| (target.selection_digest, nodes.to_vec()))
            .collect();
        let profile = derive(constraint, mesh, mixed, &prepared).unwrap();
        assert_eq!(
            original,
            prepared
                .selections()
                .map(|(target, nodes)| (target.selection_digest, nodes.to_vec()))
                .collect::<Vec<_>>()
        );
        let (compiled, _) = topology.lower(model, constraint).unwrap();
        assert_eq!(compiled.rows.len(), profile.rows.retained.row_slots);
        let terms: usize = compiled.rows.iter().map(|row| row.terms.capacity()).sum();
        assert!(
            terms <= profile.rows.retained.term_slots_upper_bound,
            "{}: {terms}",
            constraint.constraint_id
        );
        assert!(profile.total_requested_bytes().unwrap() >= profile.rows.total_requested_bytes);
    }
}

#[test]
fn original_solid_explicit_and_mixed_lowering_fit_prepared_inventory() {
    for (_, model, mesh) in crate::constraints::compilation_tests::cases() {
        check(&model, mesh.as_ref());
    }
}

#[test]
fn both_coupling_kinds_and_surface_directions_preserve_original_row_shape() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = crate::constraints::compilation_tests::mesh(order);
        let mut model =
            crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
        let coupling = ReferencePointCoupling {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
            reference_point: ReferencePoint {
                reference_point_id: "original-reference".into(),
                position: Position::new([0.2, 0.3, -1.0]),
            },
        };
        for kind in [
            StructuralConstraintKind::RigidCoupling(coupling.clone()),
            StructuralConstraintKind::DistributingCoupling(coupling.clone()),
            StructuralConstraintKind::Symmetry {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
            },
            StructuralConstraintKind::Antisymmetry {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
            },
        ] {
            model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: "original-lowering".into(),
                frame: CoordinateFrame::Global,
                kind,
            })];
            check(&model, Some(&mesh));
        }
    }
}

#[test]
fn original_equation_occurrences_and_periodic_transforms_fit_before_expansion() {
    use runmat_analysis_core::{
        Displacement, PairedConstraint, PairingPolicy, PeriodicConstraint, PeriodicTransform,
        Rotation, StructuralConstraintEquation, StructuralConstraintEquationTerm,
        StructuralTranslationDof, Translation,
    };
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet4);
    let mut model =
        crate::fixtures::fixture_model(crate::fixtures::FixtureId::CantileverLinearStatic);
    let equation = StructuralConstraintKind::Equation(StructuralConstraintEquation {
        terms: [1.0, -0.5]
            .into_iter()
            .map(|coefficient| StructuralConstraintEquationTerm {
                target: RegionTarget::point("node:1"),
                dof: StructuralTranslationDof::Ux,
                coefficient,
            })
            .collect(),
        right_hand_side: Displacement::new(0.0),
    });
    let translated = StructuralConstraintKind::Periodic(PeriodicConstraint {
        pair: PairedConstraint {
            primary: RegionTarget::point("node:1"),
            secondary: RegionTarget::point("node:2"),
            pairing: PairingPolicy::Matching,
        },
        transform: PeriodicTransform::Translation(Translation::new([1.0, 0.0, 0.0])),
    });
    let rotated = StructuralConstraintKind::Periodic(PeriodicConstraint {
        pair: PairedConstraint {
            primary: RegionTarget::point("node:3"),
            secondary: RegionTarget::point("node:4"),
            pairing: PairingPolicy::Matching,
        },
        transform: PeriodicTransform::Rotation {
            origin: Position::new([0.0; 3]),
            axis: [1.0, 0.0, 0.0],
            angle: Rotation::new(std::f64::consts::FRAC_PI_2),
        },
    });
    for kind in [equation, translated, rotated] {
        model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "original-occurrences".into(),
            frame: CoordinateFrame::Global,
            kind,
        })];
        check(&model, Some(&mesh));
    }
}

#[test]
fn cancelled_profile_does_not_consume_prepared_selections() {
    let (_, model, mesh) = crate::constraints::compilation_tests::cases().remove(0);
    let mesh = mesh.as_ref().unwrap();
    let constraint = model.constraints[0].structural().unwrap();
    let prepared = PreparedTargets::derive(constraint, Some(mesh), None).unwrap();
    let count = prepared.selections().len();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        derive(constraint, Some(mesh), None, &prepared).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(prepared.selections().len(), count);
}
