mod header;
mod initial_conditions;
mod lower;
mod requirements;
mod values;

use super::CheckedFeaDeclaration;
use crate::{
    BoundFeaProgram, CheckedExpressionKind, FeaDiagnostic, FeaSymbolKey, FeaSymbolKind,
    FeaSymbolOrigin, LogicalSourcePath,
};
use runmat_analysis_core::{AnalysisSequence, AuthoredModel};
use std::collections::BTreeMap;

pub fn lower_analysis_sequences(
    bound: &BoundFeaProgram,
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    models: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AnalysisSequence> {
    declarations
        .iter()
        .filter(|((_, key), _)| key.kind == FeaSymbolKind::Study)
        .filter_map(|(identity, declaration)| {
            let model = study_model(bound, declaration, models, diagnostics)?;
            lower::lower_study(declaration, model, diagnostics)
                .map(|sequence| (identity.clone(), sequence))
        })
        .collect()
}

fn study_model<'a>(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    models: &'a BTreeMap<(LogicalSourcePath, FeaSymbolKey), AuthoredModel>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a AuthoredModel> {
    let Some(expression) = values::assignment(&declaration.statements, "model") else {
        return missing_model(declaration, diagnostics);
    };
    let path = match &expression.kind {
        CheckedExpressionKind::Reference(path)
        | CheckedExpressionKind::FunctionCall { function: path, .. } => path,
        _ => return missing_model(declaration, diagnostics),
    };
    let [namespace, name] = path.as_slice() else {
        return missing_model(declaration, diagnostics);
    };
    if namespace != "model" {
        return missing_model(declaration, diagnostics);
    }
    let symbol_key = FeaSymbolKey {
        kind: FeaSymbolKind::Model,
        name: name.clone(),
    };
    let symbol = bound
        .symbols
        .get(&declaration.module)
        .and_then(|symbols| symbols.get(&symbol_key));
    let key = match symbol.map(|symbol| &symbol.origin) {
        Some(FeaSymbolOrigin::Local) => (declaration.module.clone(), symbol_key),
        Some(FeaSymbolOrigin::Import {
            target_module,
            target,
        }) => (target_module.clone(), target.clone()),
        _ => return missing_model(declaration, diagnostics),
    };
    models
        .get(&key)
        .or_else(|| missing_model(declaration, diagnostics))
}

fn missing_model<'a>(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a AuthoredModel> {
    super::diagnostic(
        declaration,
        declaration.span,
        "study requires one available typed `model` binding",
        diagnostics,
    );
    None
}
