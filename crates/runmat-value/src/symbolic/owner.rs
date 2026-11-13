//! Immutable original-generation symbolic metadata owners. Retaining an
//! expression never duplicates a symbolic subtree or arbitrary-precision digits.
use runmat_execution::resource::ExactShared;
use std::ops::Deref;

#[derive(Debug)]
pub struct SymbolicOwner<T>(ExactShared<T>);
impl<T> SymbolicOwner<T> {
    /// Ordinary construction moves the original node/digit allocation intact.
    pub fn new_unbounded(value: T) -> Self {
        Self(ExactShared::new(value))
    }
    pub fn shares_generation_with(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}
impl<T: Clone> SymbolicOwner<T> {
    /// Actual ordinary symbolic computation may request an owned operand.
    /// Retention/clone never invokes this materialization operation.
    pub fn into_inner_unbounded(self) -> T {
        match self.0.try_unwrap() {
            Ok(value) => value,
            Err(shared) => shared.get().clone(),
        }
    }
}
impl<T> AsRef<T> for SymbolicOwner<T> {
    fn as_ref(&self) -> &T {
        self.0.get()
    }
}
impl<T> Clone for SymbolicOwner<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> Deref for SymbolicOwner<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.as_ref()
    }
}
impl<T: PartialEq> PartialEq for SymbolicOwner<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl<T: Eq> Eq for SymbolicOwner<T> {}
impl<T> From<T> for SymbolicOwner<T> {
    fn from(value: T) -> Self {
        Self::new_unbounded(value)
    }
}
impl<T: std::fmt::Display> std::fmt::Display for SymbolicOwner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_ref().fmt(f)
    }
}
