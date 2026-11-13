//! Decode and validate the saved record before constructing the live solve.
//! The completed reader and its buffers retire before any restored callback.

use super::{error, record, session, ScalarCooperation, SolverTiming, Value};
use crate::builtins::io::mat::records::RecordPath;
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use std::path::PathBuf;

pub(super) struct Restoration {
    pub read_path: RecordPath,
    pub runtime: RuntimeContext,
    pub saved: record::Saved,
    pub overrides: Value,
}

pub(super) async fn read(
    path_value: Value,
    overrides: Value,
    c: &mut ScalarCooperation,
) -> BuiltinResult<(Restoration, &mut ScalarCooperation)> {
    let _ = c
        .runtime()
        .ok_or_else(SolverTiming::unavailable)?
        .service_ports()
        .timing()?;
    let path_text = crate::builtins::common::validation::value_text_scalar_string(&path_value)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| error("checkpoint path must be nonempty scalar text"))?;
    let read_path = session::path(PathBuf::from(path_text), c)?;
    let bytes = {
        let (runtime, budget) = c.runtime_budget();
        match (runtime, budget) {
            (Some(runtime), Some(budget)) => read_path.read(runtime, budget).await?,
            _ => {
                return Err(error(
                    "checkpoint reading lacks original cooperation authority",
                ))
            }
        }
    };
    let runtime = c
        .runtime()
        .cloned()
        .ok_or_else(|| error("checkpoint restoration requires its current original runtime"))?;
    let environment = runtime
        .program_revision()
        .ok_or_else(|| error("checkpoint restoration lacks its current program environment"))?
        .environment();
    let saved = {
        let (runtime, budget) = c.runtime_budget();
        let (Some(runtime), Some(budget)) = (runtime, budget) else {
            return Err(error(
                "checkpoint decoding lacks original cooperation authority",
            ));
        };
        let mut cooperation = super::super::json::Cooperation::new(runtime, budget)?;
        record::Saved::decode(bytes, &environment, &mut cooperation)
            .await
            .map_err(super::super::codec_error)?
    };
    if saved.version != record::VERSION || saved.environment != environment {
        return Err(error(
            "checkpoint version or execution environment is incompatible",
        ));
    }
    Ok((
        Restoration {
            read_path,
            runtime,
            saved,
            overrides,
        },
        c,
    ))
}
