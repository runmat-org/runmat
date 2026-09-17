mod header;
mod lower;
mod requirements;
mod values;

use super::CheckedFeaDeclaration;
use crate::{FeaDiagnostic, FeaSymbolKey, FeaSymbolKind, LogicalSourcePath};
use runmat_analysis_core::AnalysisSequence;
use std::collections::BTreeMap;

pub fn lower_analysis_sequences(
    declarations: &BTreeMap<(LogicalSourcePath, FeaSymbolKey), CheckedFeaDeclaration>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> BTreeMap<(LogicalSourcePath, FeaSymbolKey), AnalysisSequence> {
    declarations
        .iter()
        .filter(|((_, key), _)| key.kind == FeaSymbolKind::Study)
        .filter_map(|(identity, declaration)| {
            lower::lower_study(declaration, diagnostics)
                .map(|sequence| (identity.clone(), sequence))
        })
        .collect()
}
