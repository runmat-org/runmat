use crate::catalog::inference::{argument_error, literal_text};
use runmat_types::*;
pub(super) fn validate(
    request: &CallRequest,
    start: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !request
        .arguments
        .len()
        .saturating_sub(start)
        .is_multiple_of(2)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-CORR-OPTIONS",
            "corr options require name-value pairs",
            start,
        ));
        return;
    }
    let mut weights = false;
    for index in (start..request.arguments.len()).step_by(2) {
        let Some(name) = request
            .literals
            .literal_args
            .get(index)
            .and_then(literal_text)
        else {
            continue;
        };
        let allowed: &[&str] = match name.to_ascii_lowercase().as_str() {
            "type" => &["pearson", "spearman", "kendall"],
            "rows" => &["all", "complete", "pairwise"],
            "tail" => &["both", "right", "left"],
            "weights" => {
                if weights {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-CORR-WEIGHTS",
                        "Weights can be specified only once",
                        index,
                    ));
                }
                weights = true;
                continue;
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CORR-OPTION",
                    format!("unsupported corr option '{name}'"),
                    index,
                ));
                continue;
            }
        };
        if let Some(value) = request
            .literals
            .literal_args
            .get(index + 1)
            .and_then(literal_text)
        {
            if !allowed
                .iter()
                .any(|allowed| value.eq_ignore_ascii_case(allowed))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-CORR-OPTION-VALUE",
                    format!("unsupported {name} value '{value}'"),
                    index + 1,
                ));
            }
        }
    }
}
