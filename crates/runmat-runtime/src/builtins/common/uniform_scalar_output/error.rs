/// Classification and an optional typed operational cause live inline.
/// Constructors keep exactly one side populated; no error-path allocation is
/// introduced merely to erase the operational error's larger layout.
#[derive(Debug)]
pub(crate) struct UniformScalarError {
    failure: Option<UniformScalarFailure>,
    execution: Option<crate::RuntimeError>,
}
#[derive(Debug)]
pub(crate) enum UniformScalarFailure {
    Heterogeneous,
    NonScalar,
    InvalidStorage(&'static str),
    Materialization(String),
    CharacterRank,
    CharacterCount,
    SizeOverflow,
}
impl UniformScalarError {
    const fn classified(failure: UniformScalarFailure) -> Self {
        Self {
            failure: Some(failure),
            execution: None,
        }
    }
    pub(crate) const HETEROGENEOUS: Self = Self::classified(UniformScalarFailure::Heterogeneous);
    pub(crate) const NON_SCALAR: Self = Self::classified(UniformScalarFailure::NonScalar);
    pub(crate) const CHARACTER_RANK: Self = Self::classified(UniformScalarFailure::CharacterRank);
    pub(crate) const CHARACTER_COUNT: Self = Self::classified(UniformScalarFailure::CharacterCount);
    pub(crate) const SIZE_OVERFLOW: Self = Self::classified(UniformScalarFailure::SizeOverflow);
    pub(crate) fn invalid_storage(kind: &'static str) -> Self {
        Self::classified(UniformScalarFailure::InvalidStorage(kind))
    }
    pub(crate) fn materialization(detail: String) -> Self {
        Self::classified(UniformScalarFailure::Materialization(detail))
    }
    pub(crate) fn execution(error: crate::RuntimeError) -> Self {
        Self {
            failure: None,
            execution: Some(error),
        }
    }
    pub(crate) fn into_failure(self) -> Result<UniformScalarFailure, crate::RuntimeError> {
        match self.execution {
            Some(error) => Err(error),
            None => Ok(self.failure.expect("classification error has its failure")),
        }
    }
}
