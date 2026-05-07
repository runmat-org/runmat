pub mod lints;
pub mod schema;

pub use lints::data_api::{lint_data_api, lint_data_api_with_provider};
pub use lints::shape::lint_shapes;

pub fn lint_mir_analysis(result: &runmat_hir::LoweringResult) -> Vec<runmat_hir::HirDiagnostic> {
    let Ok(mir) = runmat_mir::lowering::lower_assembly(&result.assembly) else {
        return Vec::new();
    };
    runmat_mir::analysis::analyze_assembly(&mir).diagnostics
}
