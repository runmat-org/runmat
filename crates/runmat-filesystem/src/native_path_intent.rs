//! Portable access intent for a provider-issued native path. The intent does
//! not itself grant native path authority on any target.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePathIntent {
    Read,
    /// The consumer may open an existing file for mutation or create it.
    ReadWriteCreate,
}
