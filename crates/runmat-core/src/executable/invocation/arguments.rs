//! Distinguish source embedding input from a live, already-rooted invocation.
use runmat_runtime::call::{BuiltinArguments, FunctionArgumentIter, FunctionArgumentView};
use runmat_runtime::context::RuntimeContext;
use runmat_value::Value;

#[derive(Clone, Debug)]
pub enum ProcedureArguments {
    Source(Vec<Value>),
    Retained(BuiltinArguments),
}

impl From<Vec<Value>> for ProcedureArguments {
    fn from(values: Vec<Value>) -> Self {
        Self::Source(values)
    }
}
impl From<BuiltinArguments> for ProcedureArguments {
    fn from(values: BuiltinArguments) -> Self {
        Self::Retained(values)
    }
}
impl ProcedureArguments {
    pub fn view(&self) -> FunctionArgumentView<'_> {
        match self {
            Self::Source(values) => FunctionArgumentView::from_slice(values),
            Self::Retained(values) => values.view(),
        }
    }
    pub fn len(&self) -> usize {
        self.view().len()
    }
    pub fn is_empty(&self) -> bool {
        self.view().is_empty()
    }
    pub fn iter(&self) -> FunctionArgumentIter<'_> {
        self.view().iter()
    }

    pub(super) fn admit(
        self,
        runtime: &RuntimeContext,
    ) -> Result<BuiltinArguments, runmat_runtime::RuntimeError> {
        match self {
            Self::Source(values) => {
                runmat_runtime::execution::RootedValues::in_context(values, runtime).map(Into::into)
            }
            Self::Retained(values) => Ok(values),
        }
    }
}
