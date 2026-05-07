pub mod lints;
pub mod schema;

pub use lints::data_api::{lint_data_api, lint_data_api_with_provider};
pub use lints::shape::lint_shapes;

pub fn lint_mir_analysis(result: &runmat_hir::LoweringResult) -> Vec<runmat_hir::HirDiagnostic> {
    let mir = match runmat_mir::lowering::lower_assembly(&result.assembly) {
        Ok(mir) => mir,
        Err(err) => {
            return vec![runmat_hir::HirDiagnostic::new(
                "lint.mir.lowering_failed",
                runmat_hir::HirDiagnosticSeverity::Error,
                format!("MIR lowering failed: {}", err.message),
                err.span.unwrap_or(runmat_hir::Span { start: 0, end: 0 }),
            )
            .with_category("mir-lowering")]
        }
    };
    runmat_mir::analysis::analyze_assembly(&mir).diagnostics
}
