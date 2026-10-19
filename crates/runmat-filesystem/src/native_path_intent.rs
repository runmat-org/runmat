//! Portable access intent for a provider-issued native path. The intent does
//! not itself grant native path authority on any target.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativePathIntent {
    Read,
    /// The consumer may open an existing file for mutation or create it.
    ReadWriteCreate,
    /// The consumer may use an existing native directory as a child process's
    /// working directory. This does not authorize translating a virtual CWD.
    ProcessWorkingDirectory,
}
