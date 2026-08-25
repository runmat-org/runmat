//! Native loading is implemented in focused modules. Portable metadata users
//! never compile this module on browser/WASM targets.

mod library;

pub use library::{LoadedLibrary, LoadedSymbol, LoaderError};
