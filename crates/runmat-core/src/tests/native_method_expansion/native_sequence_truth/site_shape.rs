//! The generated-entry witnesses must actually contain the MIR sites under test.

use runmat_mir::{MirBody, MirExpressionStep, MirRvalue, MirStmtKind};

fn body<'a>(unit: &'a crate::ExecutableUnit, name: &str) -> &'a MirBody {
    let (id, _) = unit
        .mir()
        .functions
        .iter()
        .find(|(_, metadata)| metadata.name.0 == name)
        .unwrap_or_else(|| panic!("missing MIR function {name}"));
    unit.mir().bodies.get(id).expect("MIR function body")
}

fn has_rvalue(body: &MirBody, wanted: impl Fn(&MirRvalue) -> bool) -> bool {
    body.blocks.iter().any(|block| {
        block.statements.iter().any(|statement| {
            let direct = match &statement.kind {
                MirStmtKind::Assign { value, .. }
                | MirStmtKind::MultiAssign { value, .. }
                | MirStmtKind::SequenceAssign { value, .. }
                | MirStmtKind::Expr(value) => wanted(value),
                _ => false,
            };
            let mut nested = false;
            statement.visit_expression_regions(|region| {
                nested |= region.steps().iter().any(
                    |step| matches!(step, MirExpressionStep::Let { value, .. } if wanted(value)),
                );
            });
            direct || nested
        })
    })
}

fn has_capture(body: &MirBody) -> bool {
    body.blocks.iter().any(|block| {
        block.statements.iter().any(|statement| {
            let direct = matches!(&statement.kind, MirStmtKind::CaptureSequence { .. });
            let mut nested = false;
            statement.visit_expression_regions(|region| {
                nested |= region
                    .steps()
                    .iter()
                    .any(|step| matches!(step, MirExpressionStep::CaptureSequence { .. }));
            });
            direct || nested
        })
    })
}

pub(super) fn assert_compiled_sites(unit: &crate::ExecutableUnit) {
    assert!(has_rvalue(body(unit, "readMember"), |value| matches!(
        value,
        MirRvalue::Member { .. }
    )));
    assert!(has_rvalue(
        body(unit, "readDynamicMember"),
        |value| matches!(value, MirRvalue::DynamicMember { .. })
    ));
    assert!(has_rvalue(body(unit, "shortCircuit"), |value| matches!(
        value,
        MirRvalue::ShortCircuit { .. }
    )));
    assert!(has_capture(body(unit, "captureSequence")));
    assert!(has_capture(body(unit, "captureMismatch")));
}
