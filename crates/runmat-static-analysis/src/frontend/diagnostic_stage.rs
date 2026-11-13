//! Pipeline provenance, assigned by the producing compiler/adapter operation.
use super::FrontendAnalysis;
use runmat_hir::HirDiagnostic;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticStage {
    Syntax,
    Resolution,
    Semantic,
    MirLowering,
    Compile,
    Analysis,
    /// Legacy external diagnostic additions did not declare their producer.
    Unknown,
}

impl FrontendAnalysis {
    pub fn diagnostic_stage(&self, index: usize) -> DiagnosticStage {
        self.diagnostic_stages
            .get(index)
            .copied()
            .unwrap_or(DiagnosticStage::Unknown)
    }
    pub fn extend_diagnostics(
        &mut self,
        diagnostics: impl IntoIterator<Item = HirDiagnostic>,
        stage: DiagnosticStage,
    ) {
        self.diagnostic_stages
            .resize(self.diagnostics.len(), DiagnosticStage::Unknown);
        for diagnostic in diagnostics {
            self.diagnostics.push(diagnostic);
            self.diagnostic_stages.push(stage);
        }
    }
    /// Keep typed provenance attached while preserving the canonical diagnostic
    /// sort. Project composition may deduplicate exactly equal records.
    pub fn sort_diagnostics(&mut self, deduplicate: bool) {
        self.diagnostic_stages
            .resize(self.diagnostics.len(), DiagnosticStage::Unknown);
        let mut records: Vec<_> = std::mem::take(&mut self.diagnostics)
            .into_iter()
            .zip(std::mem::take(&mut self.diagnostic_stages))
            .collect();
        records.sort_by_key(|(d, _)| {
            (
                d.primary.source,
                d.primary.span.start,
                d.primary.span.end,
                d.code.clone(),
            )
        });
        if deduplicate {
            records.dedup_by(|left, right| left.0 == right.0);
        }
        (self.diagnostics, self.diagnostic_stages) = records.into_iter().unzip();
    }
}
