//! Retain a completed frame while projecting outputs and closure captures.
use crate::interpreter::vars::InterpreterVars;
use crate::runtime::gc::ManagedRoots;
use runmat_runtime::RuntimeError;

#[cfg(test)]
mod tests;

/// Values own GPU leases; compact GC roots protect their managed objects.
/// Output clones retain their own leases without publication or alias scans.
pub struct CompletedFrameProjection {
    values: InterpreterVars,
    _roots: ManagedRoots,
}

impl CompletedFrameProjection {
    pub fn new(values: impl Into<InterpreterVars>) -> Result<Self, RuntimeError> {
        let values = values.into();
        let roots = ManagedRoots::from_traced(&values).map_err(RuntimeError::new)?;
        Ok(Self {
            values,
            _roots: roots,
        })
    }

    pub fn values(&self) -> &InterpreterVars {
        &self.values
    }
}
