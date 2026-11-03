use anyhow::{bail, Result};
use runmat_execution_artifact::ProgramExecutionResponse;

#[derive(Clone, Copy)]
pub(super) enum ResultOrigin {
    Attached,
    Recovered,
}

pub(super) fn ensure_terminal_success(
    response: ProgramExecutionResponse,
    origin: ResultOrigin,
) -> Result<()> {
    match response {
        failure @ (ProgramExecutionResponse::Failure { .. }
        | ProgramExecutionResponse::RuntimeFailure { .. }) => {
            format_terminal_result(failure, origin)?;
            Ok(())
        }
        ProgramExecutionResponse::Success { .. }
        | ProgramExecutionResponse::GraphSuccess { .. }
        | ProgramExecutionResponse::ExternalizedSuccess { .. }
        | ProgramExecutionResponse::SpmdSuccess { .. } => Ok(()),
    }
}

pub(super) fn format_terminal_result(
    response: ProgramExecutionResponse,
    origin: ResultOrigin,
) -> Result<String> {
    match response {
        ProgramExecutionResponse::Success { outputs } => {
            Ok(serde_json::to_string_pretty(&outputs)?)
        }
        ProgramExecutionResponse::GraphSuccess { graph } => {
            // Presentation preserves the wire graph. It neither reconstructs
            // runtime Values nor grants callable or source authority.
            Ok(serde_json::to_string_pretty(&graph)?)
        }
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => Ok(format!(
            "{} externalized output(s), {} verified result object(s)",
            outputs.len(),
            result_objects.len()
        )),
        ProgramExecutionResponse::SpmdSuccess { .. } => match origin {
            ResultOrigin::Attached => {
                bail!("per-rank SPMD results require the gang execution consumer")
            }
            ResultOrigin::Recovered => {
                bail!("recovered per-rank SPMD results require the gang execution consumer")
            }
        },
        ProgramExecutionResponse::Failure { message } => bail!("{message}"),
        ProgramExecutionResponse::RuntimeFailure { failure } => bail!(
            "{}",
            crate::diagnostics::format_program_runtime_failure(failure)
                .map_err(anyhow::Error::msg)?
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::value::PayloadGraph;

    #[test]
    fn json_terminal_failure_keeps_nonzero_exit_semantics() {
        for origin in [ResultOrigin::Attached, ResultOrigin::Recovered] {
            let error = ensure_terminal_success(
                ProgramExecutionResponse::Failure {
                    message: "original user failure".into(),
                },
                origin,
            )
            .unwrap_err();
            assert_eq!(error.to_string(), "original user failure");
        }
    }

    #[test]
    fn terminal_results_preserve_direct_and_rooted_graph_representations() {
        for origin in [ResultOrigin::Attached, ResultOrigin::Recovered] {
            for (response, expected) in [
                (
                    ProgramExecutionResponse::Success { outputs: vec![] },
                    serde_json::json!([]),
                ),
                (
                    ProgramExecutionResponse::GraphSuccess {
                        graph: PayloadGraph {
                            roots: vec![],
                            nodes: vec![],
                        },
                    },
                    serde_json::json!({"roots": [], "nodes": []}),
                ),
            ] {
                let rendered = format_terminal_result(response, origin).unwrap();
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&rendered).unwrap(),
                    expected
                );
            }
        }
    }
}
