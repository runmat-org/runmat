use std::collections::BTreeMap;

use runmat_analysis_core::{
    AuthoredExecutionPolicy, AuthoredExecutionPolicyRef, AuthoredStudy, AuthoredSweep,
};

use crate::{
    BoundFeaProgram, CheckedExpressionKind, CheckedFeaDeclaration, FeaDiagnostic, FeaSymbolKey,
    FeaSymbolKind, LogicalSourcePath,
};

pub(super) fn bind_execution_policies(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    policies: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredExecutionPolicy>,
    studies: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredStudy>,
    sweeps: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredSweep>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredExecutionPolicyRef> {
    let mut bindings = BTreeMap::new();
    for (key, declaration) in declarations {
        if !matches!(key.1.kind, FeaSymbolKind::Study | FeaSymbolKind::Sweep) {
            continue;
        }
        let Some(value) = super::execution_policy::top_assignment(declaration, "execution") else {
            continue;
        };
        let CheckedExpressionKind::Reference(path) = &value.kind else {
            binding_error(
                declaration,
                value.span,
                "execution must reference execution.<name>",
                diagnostics,
            );
            continue;
        };
        let Some(resolved) = super::procedure::resolve_symbol_key(
            bound,
            declaration,
            path,
            FeaSymbolKind::Execution,
        ) else {
            binding_error(
                declaration,
                value.span,
                "execution policy is unavailable",
                diagnostics,
            );
            continue;
        };
        let Some(policy) = policies.get(&resolved) else {
            binding_error(
                declaration,
                value.span,
                "execution policy is invalid",
                diagnostics,
            );
            continue;
        };
        let reference = AuthoredExecutionPolicyRef {
            module_path: policy.module_path.clone(),
            policy_id: policy.policy_id.clone(),
        };
        let target_exists = match key.1.kind {
            FeaSymbolKind::Study => studies.contains_key(key),
            FeaSymbolKind::Sweep => sweeps.contains_key(key),
            _ => false,
        };
        if target_exists {
            bindings.insert(key.clone(), reference);
        }
    }
    for (sweep_key, sweep) in sweeps {
        if !bindings.contains_key(sweep_key) {
            let source_key = studies
                .iter()
                .find_map(|(study_key, study)| {
                    (study.module_path == sweep.study.module_path
                        && study.study_id == sweep.study.study_id)
                        .then_some(study_key)
                })
                .cloned();
            if let Some(reference) = source_key.and_then(|key| bindings.get(&key).cloned()) {
                bindings.insert(sweep_key.clone(), reference);
            }
        }
    }
    bindings
}

fn binding_error(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    super::execution_policy::error_at(declaration, span, message, diagnostics);
}
