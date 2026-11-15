//! Why a field scene build failed. A run without persisted render topology has its own code,
//! so a host can show the run's saved figures instead without reading the message text.

/// The code of a field scene refused because the run saved no render topology.
pub const RENDER_TOPOLOGY_MISSING_CODE: &str = "RM.FEA.RENDER_TOPOLOGY_MISSING";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldSceneError {
    /// The run's record holds no render tiles, so no field scene can be drawn from it. A
    /// rebuild of the same run fails the same way.
    RenderTopologyMissing { run_id: String },
    /// Any other failure, described by its message.
    Failed(String),
}

impl FieldSceneError {
    pub fn render_topology_missing(run_id: &str) -> Self {
        Self::RenderTopologyMissing {
            run_id: run_id.to_owned(),
        }
    }

    /// The stable code of a coded failure.
    pub fn code(&self) -> Option<&'static str> {
        match self {
            Self::RenderTopologyMissing { .. } => Some(RENDER_TOPOLOGY_MISSING_CODE),
            Self::Failed(_) => None,
        }
    }

    /// The runtime diagnostic a host renders for a coded failure (identifier, message and the
    /// run id), in the shape of the other coded FEA refusals; `None` for an uncoded failure.
    pub fn diagnostic(&self) -> Option<serde_json::Value> {
        let Self::RenderTopologyMissing { run_id } = self else {
            return None;
        };
        let message = self.to_string();
        Some(serde_json::json!({
            "kind": "runtime",
            "identifier": RENDER_TOPOLOGY_MISSING_CODE,
            "message": message,
            "diagnostic": message,
            "callstack": [],
            "runId": run_id,
        }))
    }
}

impl std::fmt::Display for FieldSceneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RenderTopologyMissing { run_id } => {
                write!(f, "FEA run '{run_id}' has no persisted render topology")
            }
            Self::Failed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for FieldSceneError {}

impl From<String> for FieldSceneError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

impl From<&str> for FieldSceneError {
    fn from(message: &str) -> Self {
        Self::Failed(message.to_owned())
    }
}

impl From<FieldSceneError> for String {
    fn from(error: FieldSceneError) -> Self {
        error.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_render_topology_is_a_coded_diagnostic_with_the_run_id() {
        let error = FieldSceneError::render_topology_missing("run_1");
        assert_eq!(error.code(), Some("RM.FEA.RENDER_TOPOLOGY_MISSING"));
        assert_eq!(
            error.to_string(),
            "FEA run 'run_1' has no persisted render topology"
        );
        let diagnostic = error.diagnostic().expect("coded diagnostic");
        assert_eq!(diagnostic["identifier"], RENDER_TOPOLOGY_MISSING_CODE);
        assert_eq!(diagnostic["runId"], "run_1");
        assert_eq!(diagnostic["message"], error.to_string());
        let other = FieldSceneError::from("FEA run has no renderable surface mesh");
        assert_eq!(other.code(), None);
        assert!(other.diagnostic().is_none());
    }
}
