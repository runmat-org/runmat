use runmat_mir::MirConstructKind;
use runmat_types::{FunctionId, ProgramPointId};

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct NativeCodegenError {
    pub code: &'static str,
    pub message: String,
    pub function: Option<FunctionId>,
    pub point: Option<ProgramPointId>,
    pub construct: Option<MirConstructKind>,
    #[source]
    pub allocation: Option<runmat_runtime::context::host_allocation::HostAllocationError>,
}

impl NativeCodegenError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            function: None,
            point: None,
            construct: None,
            allocation: None,
        }
    }

    pub fn at_function(mut self, function: FunctionId) -> Self {
        self.function = Some(function);
        self
    }

    pub fn with_allocation(
        mut self,
        cause: runmat_runtime::context::host_allocation::HostAllocationError,
    ) -> Self {
        self.allocation = Some(cause);
        self
    }

    pub fn at_point(mut self, point: ProgramPointId) -> Self {
        self.point = Some(point);
        self
    }

    pub fn for_construct(mut self, construct: MirConstructKind) -> Self {
        self.construct = Some(construct);
        self
    }
}

pub type NativeCodegenResult<T> = Result<T, NativeCodegenError>;
