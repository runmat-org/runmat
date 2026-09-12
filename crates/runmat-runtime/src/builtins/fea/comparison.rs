use crate::analysis::analysis_results_compare_op;
use crate::analysis::analysis_trends_op;
use crate::analysis::AnalysisResultsCompareQuery;
use crate::analysis::AnalysisTrendsQuery;
use crate::builtins::fea::contracts::descriptors::{ERROR_INPUT, ERROR_INTERNAL, ERROR_OPERATION};
use crate::builtins::fea::contracts::identities::{
    COMPARE_NAME, FEA_COMPARE_CLASS, FEA_PAYLOAD_JSON_PROPERTY, FEA_TRENDS_CLASS, TRENDS_NAME,
};
use crate::builtins::fea::errors::builtin_error;
use crate::builtins::fea::geometry::scalar_string;
use crate::builtins::fea::options_json::expect_name_value_tail;
use crate::builtins::fea::output::{
    operation_result_to_object, operation_result_to_object_preserving_integers,
};
use crate::builtins::fea::value_decode::usize_from_value;
use crate::operations::OperationContext;
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::fea) fn create_compare_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.len() != 2 {
        return Err(builtin_error(
            COMPARE_NAME,
            &ERROR_INPUT,
            "fea.compare requires baseline and candidate run ids",
        ));
    }
    let baseline_run_id = scalar_string(&args[0], COMPARE_NAME, &ERROR_INPUT)?;
    let candidate_run_id = scalar_string(&args[1], COMPARE_NAME, &ERROR_INPUT)?;
    operation_result_to_object_preserving_integers(
        COMPARE_NAME,
        &ERROR_OPERATION,
        &ERROR_INTERNAL,
        FEA_COMPARE_CLASS,
        analysis_results_compare_op(
            AnalysisResultsCompareQuery {
                baseline_run_id,
                candidate_run_id,
            },
            OperationContext::new(None, None),
        ),
        Some(FEA_PAYLOAD_JSON_PROPERTY),
        &[
            "quality_reason_count_delta",
            "failed_increment_delta",
            "max_iteration_delta",
            "nonlinear_spike_count_delta",
            "nonlinear_stall_count_delta",
        ],
        &[],
    )
}

pub(in crate::builtins::fea) fn create_trends_object_from_args(
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let mut window_size = AnalysisTrendsQuery::default().window_size;
    let mut window_size_seen = false;
    for pair in expect_name_value_tail(TRENDS_NAME, args.as_slice())? {
        match pair.key.as_str() {
            "windowsize" => {
                if window_size_seen {
                    return Err(builtin_error(
                        TRENDS_NAME,
                        &ERROR_INPUT,
                        "duplicate fea.trends option `windowsize`",
                    ));
                }
                window_size_seen = true;
                window_size = usize_from_value(TRENDS_NAME, pair.value)?;
                if window_size == 0 {
                    return Err(builtin_error(
                        TRENDS_NAME,
                        &ERROR_INPUT,
                        "fea.trends WindowSize must be positive",
                    ));
                }
            }
            other => {
                return Err(builtin_error(
                    TRENDS_NAME,
                    &ERROR_INPUT,
                    format!("unsupported fea.trends option `{other}`"),
                ));
            }
        }
    }
    operation_result_to_object(
        TRENDS_NAME,
        &ERROR_OPERATION,
        &ERROR_INTERNAL,
        FEA_TRENDS_CLASS,
        analysis_trends_op(
            AnalysisTrendsQuery { window_size },
            OperationContext::new(None, None),
        ),
        Some(FEA_PAYLOAD_JSON_PROPERTY),
    )
}
