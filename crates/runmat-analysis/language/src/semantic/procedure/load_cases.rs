//! Procedure-local load selection, independent of constraint/object activation.
use super::lower::error;
use crate::{
    CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic,
};
use runmat_analysis_core::{AuthoredModel, ExcitationCaseId, ExcitationCaseSelection};
use runmat_design_syntax::FeaAssignmentRelation;
use std::collections::BTreeSet;

pub(super) fn is_selection(statement: &CheckedFeaStatement) -> bool {
    matches!(&statement.kind, CheckedFeaStatementKind::Assignment { target, .. }
        if target.first().is_some_and(|name| name == "load_cases"))
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    models: &[&AuthoredModel],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ExcitationCaseSelection> {
    let mut assignments = statements
        .iter()
        .filter(|statement| is_selection(statement));
    let Some(statement) = assignments.next() else {
        return Some(ExcitationCaseSelection::All);
    };
    if assignments.next().is_some() {
        error(
            declaration,
            statement.span,
            "load_cases must be assigned exactly once per procedure",
            diagnostics,
        );
        return None;
    }
    let CheckedFeaStatementKind::Assignment {
        target,
        relation: FeaAssignmentRelation::Equal,
        value,
    } = &statement.kind
    else {
        error(
            declaration,
            statement.span,
            "load_cases requires an = assignment",
            diagnostics,
        );
        return None;
    };
    let CheckedExpressionKind::List(values) = &value.kind else {
        error(
            declaration,
            value.span,
            "load_cases requires a list of load_case.<name> references",
            diagnostics,
        );
        return None;
    };
    if target.len() != 1 {
        error(
            declaration,
            statement.span,
            "load_cases does not accept nested properties",
            diagnostics,
        );
        return None;
    }
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();
    for value in values {
        let CheckedExpressionKind::Reference(path) = &value.kind else {
            error(
                declaration,
                value.span,
                "load_cases requires load_case.<name> references",
                diagnostics,
            );
            return None;
        };
        let [namespace, name] = path.as_slice() else {
            error(
                declaration,
                value.span,
                "load_cases requires load_case.<name> references",
                diagnostics,
            );
            return None;
        };
        let id = format!("{namespace}.{name}");
        if namespace != "load_case" || !seen.insert(id.clone()) {
            error(
                declaration,
                value.span,
                "load_cases entries must be unique load_case.<name> references",
                diagnostics,
            );
            return None;
        }
        let count = models
            .iter()
            .flat_map(|model| &model.excitation_cases)
            .filter(|case| case.case_id == id)
            .count();
        if count != 1 {
            error(
                declaration,
                value.span,
                format!("load case `{id}` must resolve uniquely in the study model"),
                diagnostics,
            );
            return None;
        }
        selected.push(ExcitationCaseId(id));
    }
    Some(ExcitationCaseSelection::Only(selected))
}
