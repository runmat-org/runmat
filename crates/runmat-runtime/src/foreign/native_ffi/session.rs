use std::collections::BTreeMap;
use std::rc::Rc;

use runmat_native_ffi::{
    LoadedLibrary, NativeLibraryMetadata, NativePointer, NativePointerResource,
};

#[derive(Clone)]
pub(super) struct LibraryEntry {
    pub library: Rc<LoadedLibrary>,
    pub metadata: Rc<NativeLibraryMetadata>,
    pub artifact_identity: String,
}

impl std::fmt::Debug for LibraryEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LibraryEntry")
            .field("metadata", &self.metadata)
            .field("artifact_identity", &self.artifact_identity)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(super) enum PointerEntry {
    CallerOwned {
        pointer: Rc<NativePointerResource>,
        metadata: Rc<NativeLibraryMetadata>,
        type_name: String,
        library: Option<Rc<LoadedLibrary>>,
    },
    Opaque {
        pointer: Rc<NativePointer>,
        library: Rc<LoadedLibrary>,
        metadata: Rc<NativeLibraryMetadata>,
    },
}

impl std::fmt::Debug for PointerEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CallerOwned {
                pointer,
                type_name,
                library,
                ..
            } => formatter
                .debug_struct("CallerOwned")
                .field("pointer", pointer)
                .field("type_name", type_name)
                .field(
                    "library_references",
                    &library.as_ref().map(Rc::strong_count),
                )
                .finish(),
            Self::Opaque {
                pointer,
                library,
                metadata,
            } => formatter
                .debug_struct("Opaque")
                .field("pointer", pointer)
                .field("library_references", &Rc::strong_count(library))
                .field("target_triple", &metadata.target_triple)
                .finish(),
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct NativeFfiSessionState {
    pub libraries: BTreeMap<String, LibraryEntry>,
    pub pointers: BTreeMap<u64, PointerEntry>,
}
