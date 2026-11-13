mod compare;
mod observation_history;
mod query;
pub use observation_history::analysis_observation_history_sample_by_run_id_op;
mod trends;

use super::{diagnostic_metric, AnalysisRunResult};

pub use compare::analysis_results_compare_op;
pub use query::{
    analysis_field_page_by_run_id_op, analysis_results_by_run_id_op, analysis_results_op,
    bounded_default_results_query, RESULTS_INLINE_FIELD_BYTES,
};
pub(crate) use query::{
    analysis_run_field_ids, collect_analysis_result_fields, visit_analysis_result_fields,
    visit_analysis_result_fields_mut,
};
pub use trends::analysis_trends_op;

fn run_solve_ms(run: &AnalysisRunResult) -> Option<f64> {
    for code in [
        "FEA_NONLINEAR_COST",
        "FEA_TRANSIENT_COST",
        "FEA_MODAL_COST",
        "FEA_ACOUSTIC_COST",
        "FEA_CFD_COST",
        "FEA_CHT_COST",
        "FEA_FSI_COST",
    ] {
        if let Some(value) = diagnostic_metric(&run.run.diagnostics, code, "solve_ms") {
            return Some(value);
        }
    }
    None
}
