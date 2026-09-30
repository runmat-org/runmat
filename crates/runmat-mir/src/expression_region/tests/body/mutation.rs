use super::*;

#[test]
fn mutation_annotation_does_not_duplicate_the_store_region() {
    let region = MirExpressionRegion {
        steps: vec![let_step(1, MirRvalue::End)],
        result: MirOperand::Local(MirLocalId(1)),
    };
    let mut body = body_with_region(region);
    let place = match &body.blocks[0].statements[0].kind {
        MirStmtKind::Assign { place, .. } => place.clone(),
        other => panic!("expected assignment, got {other:?}"),
    };
    body.blocks[0].statements.insert(
        0,
        MirStmt {
            kind: MirStmtKind::PlaceMutation(crate::MirPlaceMutation {
                place,
                kind: runmat_hir::PlaceMutationKind::IndexedAssign,
                creation_policy: runmat_hir::AssignmentCreationPolicy::ExistingOnly,
                shape_policy: runmat_hir::AssignmentShapePolicy::MatlabCompatible,
            }),
            span: Span::default(),
        },
    );
    body.validate_expression_regions().unwrap();

    let mut missing_store = body.clone();
    missing_store.blocks[0].statements.remove(1);
    assert!(missing_store
        .validate_expression_regions()
        .unwrap_err()
        .contains("no following assignment"));

    let mut mismatched_store = body;
    let MirStmtKind::Assign { place, .. } = &mut mismatched_store.blocks[0].statements[1].kind
    else {
        unreachable!()
    };
    *place = crate::MirPlace::Local(MirLocalId(9));
    assert!(mismatched_store
        .validate_expression_regions()
        .unwrap_err()
        .contains("exact assignment"));
}
