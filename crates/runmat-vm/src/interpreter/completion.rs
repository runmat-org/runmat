//! Retain a completed frame while projecting outputs and closure captures.
use crate::runtime::gc::ManagedRoots;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

#[cfg(test)]
mod tests;

/// Values own GPU leases; compact GC roots protect their managed objects.
/// Output clones retain their own leases without publication or alias scans.
pub struct CompletedFrameProjection {
    values: Vec<Value>,
    _roots: ManagedRoots,
}

impl CompletedFrameProjection {
    pub fn new(values: Vec<Value>) -> Result<Self, RuntimeError> {
        let roots = ManagedRoots::new(&values).map_err(RuntimeError::new)?;
        Ok(Self {
            values,
            _roots: roots,
        })
    }

    pub fn values(&self) -> &[Value] {
        &self.values
    }
}
