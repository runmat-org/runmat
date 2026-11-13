//! DataArray numeric read, write, zero-resize, and scalar-fill contracts.

use super::super::{
    contract::{self, Behavior, FILESYSTEM, READ_EFFECTS, WRITE_EFFECTS},
    descriptor as d, documentation as docs,
    inference::{self, ArrayForm},
    integer,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

macro_rules! payload {
    ($entry:ident, $descriptor:ident, $infer:ident, $name:literal, $label:literal, $inputs:expr, $output:expr, $form:expr, $doc:expr, $effects:expr, $purity:expr, $integers:expr) => {
        d::receiver_descriptor!($descriptor, $label, $inputs, $output);
        fn $infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            inference::array_infer(request, entry, $form)
        }
        pub const $entry: BuiltinCatalogEntry = contract::entry(
            BuiltinCatalogProvenance::new(file!(), module_path!()),
            $name,
            $doc,
            &$descriptor,
            $infer,
            Behavior {
                effects: $effects,
                capabilities: FILESYSTEM,
                purity: $purity,
                async_behavior: BuiltinAsyncBehavior::MaySuspend,
                integer_capabilities: $integers,
                integer_audit: None,
            },
        );
    };
}

const PAYLOAD_VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "All integer classes and paired complex storage", location: "builtins::io::data::tests::data_arrays_preserve_every_integer_class_through_chunked_write_and_read" },
    BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Pinned provider and U64 write/read", location: "builtins::io::data::tests::data_create_retains_selected_provider_and_relative_cwd_after_first_write" },
];

const READ_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.read",
    summary: "Read a full stored array or a selected slice.",
    description: "`X = DataArray.read(arr, sliceSpec)` decodes stored chunks through the handle's pinned provider and returns a host numeric tensor; omitting the slice reads the full array.",
    keywords: &["DataArray.read", "dataset", "chunked", "slice"],
    related: &["DataArray.write", "DataArray.dtype"],
    sections: &[BuiltinDocumentationSection { heading: "Class and slice", paragraphs: &["The declared f64, f32, or integer class and real/paired-complex representation are preserved. A slice specification is a cell of one-based scalar indices, inclusive two-endpoint ranges, or ':' entries; the current implementation may decode the full payload before extracting a slice, so a small slice is not a certified bounded-memory read."] }],
    examples: &[docs::example("array-read", "Read a stored integer array", "schema = struct(); schema.arrays.sample = struct('dtype', 'uint64', 'shape', [1 2]);\nds = data.create('read.data', schema);\narr = Dataset.array(ds, 'sample');\nX = DataArray.read(arr);", "assert(isa(X, 'uint64')); assert(isequal(size(X), [1 2])); assert(all(X == uint64(0)));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Is the result resident on the GPU?", answer: "No. This dataset read returns a host tensor in the declared numeric class." }],
    verification: PAYLOAD_VERIFICATION,
});
payload!(
    DATAARRAY_READ_CATALOG_ENTRY,
    DATAARRAY_READ_DESCRIPTOR,
    infer_read,
    "DataArray.read",
    "X = DataArray.read(arr, sliceSpec)",
    &[d::ARRAY, d::SLICE_OPTIONAL],
    &[d::OUT_TENSOR],
    ArrayForm::Read,
    READ_DOC,
    READ_EFFECTS,
    BuiltinPurity::DeterministicReadOnly,
    integer::ARRAY_READ
);

const WRITE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.write",
    summary: "Write full-array or selected-slice numeric values.",
    description: "`tf = DataArray.write(arr, values)` replaces the array payload; `tf = DataArray.write(arr, sliceSpec, values)` updates a selected region through the pinned provider.",
    keywords: &["DataArray.write", "dataset", "chunks", "slice"],
    related: &["DataArray.read", "DataTransaction.write"],
    sections: &[BuiltinDocumentationSection { heading: "Typed storage", paragraphs: &["Same-class integer and paired-complex values retain exact bits. Other numeric classes are cast once to the declared dtype, and shape or complexity mismatches reject. With an existing chunk index, slice writes create fresh objects for touched chunks and share untouched chunks. The manifest publishes the replacement index; rejected publication leaves previously referenced objects unchanged and can leave unreferenced revision files."] }],
    examples: &[docs::example("array-write", "Write and read an integer array", "schema = struct(); schema.arrays.sample = struct('dtype', 'uint64', 'shape', [1 2]);\nds = data.create('write.data', schema);\narr = Dataset.array(ds, 'sample');\ntf = DataArray.write(arr, uint64([5 6]));\nX = DataArray.read(arr);", "assert(tf); assert(isequal(X, uint64([5 6])));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Is a multi-chunk update atomic?", answer: "No. A later provider or serialization failure may leave some chunk writes before manifest publication." }],
    verification: PAYLOAD_VERIFICATION,
});
// The three-argument slice form is part of the same public identity and is
// represented by a second signature, not a separate builtin entry.
pub const DATAARRAY_WRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "tf = DataArray.write(arr, values)",
            inputs: &[d::ARRAY, d::VALUES],
            outputs: &[d::OUT_BOOL],
        },
        BuiltinSignatureDescriptor {
            label: "tf = DataArray.write(arr, sliceSpec, values)",
            inputs: &[d::ARRAY, d::SLICE_REQUIRED, d::VALUES],
            outputs: &[d::OUT_BOOL],
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: super::super::descriptor::ERRORS,
};
fn infer_write(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    inference::array_infer(request, entry, ArrayForm::Write)
}
pub const DATAARRAY_WRITE_CATALOG_ENTRY: BuiltinCatalogEntry = contract::entry(
    BuiltinCatalogProvenance::new(file!(), module_path!()),
    "DataArray.write",
    WRITE_DOC,
    &DATAARRAY_WRITE_DESCRIPTOR,
    infer_write,
    Behavior {
        effects: WRITE_EFFECTS,
        capabilities: FILESYSTEM,
        purity: BuiltinPurity::Impure,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        integer_capabilities: integer::ARRAY_WRITE,
        integer_audit: None,
    },
);

const RESIZE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.resize",
    summary: "Replace an array with zero-filled storage at a new shape.",
    description: "`tf = DataArray.resize(arr, newShape)` updates the manifest shape and writes new zero-filled chunks in the declared dtype through the pinned provider.",
    keywords: &["DataArray.resize", "dataset", "shape", "zero-fill"],
    related: &["DataArray.shape", "DataArray.fill"],
    sections: &[BuiltinDocumentationSection { heading: "Destructive resize", paragraphs: &["The current implementation recreates zero-filled payload storage and does not preserve previous values. Dimensions accept exact typed integers or integral floats within nonnegative platform bounds; checked total element multiplication may reject before allocation."] }],
    examples: &[docs::example("array-resize", "Resize to zero-filled storage", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('resize.data', schema);\narr = Dataset.array(ds, 'sample');\ntf = DataArray.resize(arr, [2 2]);", "assert(tf); assert(isequal(DataArray.shape(arr), [2 2])); assert(all(DataArray.read(arr) == 0));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Are old values retained?", answer: "No. This form recreates zero-filled storage at the new shape." }],
    verification: PAYLOAD_VERIFICATION,
});
payload!(
    DATAARRAY_RESIZE_CATALOG_ENTRY,
    DATAARRAY_RESIZE_DESCRIPTOR,
    infer_resize,
    "DataArray.resize",
    "tf = DataArray.resize(arr, newShape)",
    &[d::ARRAY, d::NEW_SHAPE],
    &[d::OUT_BOOL],
    ArrayForm::Resize,
    RESIZE_DOC,
    WRITE_EFFECTS,
    BuiltinPurity::Impure,
    integer::ARRAY_RESIZE
);

const FILL_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.fill",
    summary: "Fill the complete stored array with one numeric scalar.",
    description: "`tf = DataArray.fill(arr, value)` casts one numeric scalar to the declared dtype, repeats it across the full array, and publishes a new manifest sequence.",
    keywords: &["DataArray.fill", "dataset", "scalar", "chunks"],
    related: &["DataArray.resize", "DataTransaction.fill"],
    sections: &[BuiltinDocumentationSection { heading: "Class and effects", paragraphs: &["Same-class real or complex integer scalars retain exact bits; unlike numeric classes follow the declared dtype cast. Nonscalar fill inputs reject. This operation writes a fresh full-payload revision. Rejected manifest publication preserves previously referenced objects and can leave unreferenced revision files."] }],
    examples: &[docs::example("array-fill", "Fill a typed array", "schema = struct(); schema.arrays.sample = struct('dtype', 'uint64', 'shape', [1 2]);\nds = data.create('fill.data', schema);\narr = Dataset.array(ds, 'sample');\ntf = DataArray.fill(arr, uint64(7));", "assert(tf); assert(isequal(DataArray.read(arr), uint64([7 7])));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Can this fill only a slice?", answer: "Not directly. Use `DataTransaction.fill` with an optional slice or write a slice with `DataArray.write`." }],
    verification: PAYLOAD_VERIFICATION,
});
payload!(
    DATAARRAY_FILL_CATALOG_ENTRY,
    DATAARRAY_FILL_DESCRIPTOR,
    infer_fill,
    "DataArray.fill",
    "tf = DataArray.fill(arr, value)",
    &[d::ARRAY, d::VALUE],
    &[d::OUT_BOOL],
    ArrayForm::Fill,
    FILL_DOC,
    WRITE_EFFECTS,
    BuiltinPurity::Impure,
    integer::ARRAY_FILL
);
