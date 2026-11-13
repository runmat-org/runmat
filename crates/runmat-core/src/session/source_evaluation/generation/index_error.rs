use std::error::Error;
use std::fmt;

/// Admission failures owned by the session's callable artifact index.
#[derive(Debug)]
pub(in crate::session::source_evaluation) enum CallableArtifactIndexError {
    InvalidUnit(runmat_vm::ExecutableUnitAdmissionError),
    InvalidDefinitions(runmat_vm::RetainedCallableStageError),
    ConflictingOrigin,
}

impl fmt::Display for CallableArtifactIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUnit(source) => {
                write!(formatter, "invalid callable artifact unit: {source}")
            }
            Self::InvalidDefinitions(source) => {
                write!(formatter, "invalid callable artifact definitions: {source}")
            }
            Self::ConflictingOrigin => formatter.write_str(
                "invalid callable artifact index: one immutable function origin has conflicting executable bodies",
            ),
        }
    }
}

impl Error for CallableArtifactIndexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidUnit(source) => Some(source),
            Self::InvalidDefinitions(source) => Some(source),
            Self::ConflictingOrigin => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_failures_retain_the_original_typed_admission_cause() {
        let error = CallableArtifactIndexError::InvalidUnit(
            runmat_vm::ExecutableUnitAdmissionError::MissingComponent("bytecode"),
        );
        assert!(error
            .source()
            .unwrap()
            .is::<runmat_vm::ExecutableUnitAdmissionError>());
        let error = CallableArtifactIndexError::InvalidDefinitions(
            runmat_vm::RetainedCallableStageError::IncompleteDefinitionGraph,
        );
        assert!(error
            .source()
            .unwrap()
            .is::<runmat_vm::RetainedCallableStageError>());
        assert!(CallableArtifactIndexError::ConflictingOrigin
            .source()
            .is_none());
    }
}
