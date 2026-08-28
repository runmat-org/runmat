use runmat_mir::MirConstructKind;
use runmat_types::{ProgramFunctionId, ProgramPointId};

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct NativeCodegenError {
    pub code: &'static str,
    pub message: String,
    pub function: Option<ProgramFunctionId>,
    pub point: Option<ProgramPointId>,
    pub construct: Option<MirConstructKind>,
}

impl NativeCodegenError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            function: None,
            point: None,
            construct: None,
        }
    }

    pub fn at_function(mut self, function: ProgramFunctionId) -> Self {
        self.function = Some(function);
        self
    }

    pub fn at_point(mut self, point: ProgramPointId) -> Self {
        self.function = Some(point.function);
        self.point = Some(point);
        self
    }

    pub fn for_construct(mut self, construct: MirConstructKind) -> Self {
        self.construct = Some(construct);
        self
    }
}

pub type NativeCodegenResult<T> = Result<T, NativeCodegenError>;
