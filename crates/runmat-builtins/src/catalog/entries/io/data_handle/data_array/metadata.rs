//! DataArray identity and manifest metadata accessors.

use super::super::{
    contract::{self, Behavior, ACCESS_EFFECTS, FILESYSTEM, READ_EFFECTS},
    descriptor as d, documentation as docs,
    inference::{self, ArrayForm},
    integer::ARRAY_METADATA_AUDIT,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

const LOCAL: Behavior = Behavior {
    effects: ACCESS_EFFECTS,
    capabilities: &[],
    purity: BuiltinPurity::DeterministicReadOnly,
    async_behavior: BuiltinAsyncBehavior::NeverSuspends,
    integer_capabilities: &[],
    integer_audit: Some(&ARRAY_METADATA_AUDIT),
};
const READ: Behavior = Behavior {
    effects: READ_EFFECTS,
    capabilities: FILESYSTEM,
    purity: BuiltinPurity::DeterministicReadOnly,
    async_behavior: BuiltinAsyncBehavior::MaySuspend,
    integer_capabilities: &[],
    integer_audit: Some(&ARRAY_METADATA_AUDIT),
};

macro_rules! metadata {
    ($entry:ident, $descriptor:ident, $infer:ident, $name:literal, $label:literal, $output:expr, $form:expr, $doc:expr, $behavior:expr) => {
        d::receiver_descriptor!($descriptor, $label, &[d::ARRAY], $output);
        fn $infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
            inference::array_infer(request, entry, $form)
        }
        pub const $entry: BuiltinCatalogEntry = contract::entry(
            BuiltinCatalogProvenance::new(file!(), module_path!()),
            $name,
            $doc,
            &$descriptor,
            $infer,
            $behavior,
        );
    };
}

const VERIFICATION: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Typed DataArray metadata and receiver checks",
    location: "builtins::io::data::tests::data_array_metadata_accessors_reject_integer_receivers",
}];

const NAME_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.name",
    summary: "Return a DataArray handle's sealed array name.",
    description: "`name = DataArray.name(arr)` returns the array name captured when `Dataset.array` issued the child handle; no manifest or payload is read.",
    keywords: &["DataArray.name", "dataset", "array", "name"],
    related: &["Dataset.array", "DataArray.dtype"],
    sections: &[BuiltinDocumentationSection { heading: "Handle identity", paragraphs: &["The child retains the Dataset's exact provider and resolved root. Its public name property does not authorize a different array or provider."] }],
    examples: &[docs::example("array-name", "Read a child name", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('arrayname.data', schema);\narr = Dataset.array(ds, 'sample');\nname = DataArray.name(arr);", "assert(strcmp(name, 'sample'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does this check the current manifest?", answer: "No. It presents the handle's sealed child name." }],
    verification: VERIFICATION,
});
metadata!(
    DATAARRAY_NAME_CATALOG_ENTRY,
    DATAARRAY_NAME_DESCRIPTOR,
    infer_name,
    "DataArray.name",
    "name = DataArray.name(arr)",
    &[d::OUT_STRING],
    ArrayForm::Name,
    NAME_DOC,
    LOCAL
);

const DTYPE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.dtype",
    summary: "Read an array's declared stored numeric class.",
    description: "`dtype = DataArray.dtype(arr)` rereads the manifest through the handle's pinned provider and returns the array's dtype token.",
    keywords: &["DataArray.dtype", "dataset", "array", "numeric class"],
    related: &["DataArray.read", "DataArray.shape"],
    sections: &[BuiltinDocumentationSection { heading: "Declared class", paragraphs: &["Supported stored dtypes are f64, f32, and all eight signed/unsigned fixed-width integer classes. The call reads metadata, not numeric chunks."] }],
    examples: &[docs::example("array-dtype", "Read a declared dtype", "schema = struct(); schema.arrays.sample = struct('dtype', 'uint64', 'shape', [1 1]);\nds = data.create('dtype.data', schema);\narr = Dataset.array(ds, 'sample');\ndtype = DataArray.dtype(arr);", "assert(strcmp(dtype, 'uint64'));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does dtype change an existing payload?", answer: "No. It reports the class declared in the current manifest." }],
    verification: VERIFICATION,
});
metadata!(
    DATAARRAY_DTYPE_CATALOG_ENTRY,
    DATAARRAY_DTYPE_DESCRIPTOR,
    infer_dtype,
    "DataArray.dtype",
    "dtype = DataArray.dtype(arr)",
    &[d::OUT_STRING],
    ArrayForm::DType,
    DTYPE_DOC,
    READ
);

const SHAPE_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.shape",
    summary: "Read an array's full dimension vector.",
    description: "`shape = DataArray.shape(arr)` reads the current manifest and returns a row double vector of dimensions.",
    keywords: &["DataArray.shape", "dataset", "array", "dimensions"],
    related: &["DataArray.rank", "DataArray.chunk_shape"],
    sections: &[BuiltinDocumentationSection { heading: "Metadata projection", paragraphs: &["Stored manifest dimensions are platform-sized integers; this display accessor projects them to double values, so dimensions above double's exact integer range may lose presentation precision. Structural shape controls retain checked typed integers at write boundaries."] }],
    examples: &[docs::example("array-shape", "Read array dimensions", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [2 3]);\nds = data.create('shape.data', schema);\narr = Dataset.array(ds, 'sample');\nshape = DataArray.shape(arr);", "assert(isequal(shape, [2 3]));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Are payload chunks decoded?", answer: "No. Dimensions come from the current manifest." }],
    verification: VERIFICATION,
});
metadata!(
    DATAARRAY_SHAPE_CATALOG_ENTRY,
    DATAARRAY_SHAPE_DESCRIPTOR,
    infer_shape,
    "DataArray.shape",
    "shape = DataArray.shape(arr)",
    &[d::OUT_TENSOR],
    ArrayForm::Shape,
    SHAPE_DOC,
    READ
);

const RANK_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.rank",
    summary: "Return the number of stored shape dimensions.",
    description: "`rank = DataArray.rank(arr)` reads the current manifest and returns a scalar double count of dimensions.",
    keywords: &["DataArray.rank", "dataset", "array", "dimensions"],
    related: &["DataArray.shape", "DataArray.chunk_shape"],
    sections: &[BuiltinDocumentationSection { heading: "Metadata", paragraphs: &["Rank is the length of the manifest shape vector, not the number of non-singleton dimensions. No array payload is decoded."] }],
    examples: &[docs::example("array-rank", "Read an array rank", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [2 3]);\nds = data.create('rank.data', schema);\narr = Dataset.array(ds, 'sample');\nrank = DataArray.rank(arr);", "assert(rank == 2);" )],
    faqs: &[BuiltinDocumentationFaq { question: "Is rank the count of non-singleton dimensions?", answer: "No. It is the stored shape vector's length." }],
    verification: VERIFICATION,
});
metadata!(
    DATAARRAY_RANK_CATALOG_ENTRY,
    DATAARRAY_RANK_DESCRIPTOR,
    infer_rank,
    "DataArray.rank",
    "rank = DataArray.rank(arr)",
    &[d::OUT_VALUE],
    ArrayForm::Rank,
    RANK_DOC,
    READ
);

const CHUNK_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.chunk_shape",
    summary: "Read an array's chunk dimension vector.",
    description: "`chunkShape = DataArray.chunk_shape(arr)` reads the manifest and returns the declared chunk extents as a row double vector.",
    keywords: &["DataArray.chunk_shape", "dataset", "array", "chunks"],
    related: &["DataArray.shape", "DataArray.read"],
    sections: &[BuiltinDocumentationSection { heading: "Chunk layout", paragraphs: &["Chunk extents are metadata used to partition persisted array storage. The accessor does not load or upload chunks; very large stored integer extents may lose exactness when projected to double for display."] }],
    examples: &[docs::example("array-chunk-shape", "Read chunk extents", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [4 4], 'chunk', [2 2]);\nds = data.create('chunkshape.data', schema);\narr = Dataset.array(ds, 'sample');\nchunkShape = DataArray.chunk_shape(arr);", "assert(isequal(chunkShape, [2 2]));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does this trigger an upload?", answer: "No. It reads only manifest metadata." }],
    verification: VERIFICATION,
});
metadata!(
    DATAARRAY_CHUNK_SHAPE_CATALOG_ENTRY,
    DATAARRAY_CHUNK_SHAPE_DESCRIPTOR,
    infer_chunk_shape,
    "DataArray.chunk_shape",
    "chunkShape = DataArray.chunk_shape(arr)",
    &[d::OUT_TENSOR],
    ArrayForm::ChunkShape,
    CHUNK_DOC,
    READ
);

const CODEC_DOC: BuiltinDocumentation = docs::documentation(docs::Content {
    name: "DataArray.codec",
    summary: "Read an array's declared storage codec.",
    description: "`codec = DataArray.codec(arr)` reads the current manifest and returns the codec token recorded for the array.",
    keywords: &["DataArray.codec", "dataset", "array", "compression"],
    related: &["DataArray.dtype", "DataArray.read"],
    sections: &[BuiltinDocumentationSection { heading: "Storage metadata", paragraphs: &["The token describes the manifest entry; this accessor does not decompress or read array chunks."] }],
    examples: &[docs::example("array-codec", "Read a codec token", "schema = struct(); schema.arrays.sample = struct('dtype', 'f64', 'shape', [1 1]);\nds = data.create('codec.data', schema);\narr = Dataset.array(ds, 'sample');\ncodec = DataArray.codec(arr);", "assert(~isempty(codec));" )],
    faqs: &[BuiltinDocumentationFaq { question: "Does codec change the stored array?", answer: "No. It is a read-only manifest accessor." }],
    verification: VERIFICATION,
});
metadata!(
    DATAARRAY_CODEC_CATALOG_ENTRY,
    DATAARRAY_CODEC_DESCRIPTOR,
    infer_codec,
    "DataArray.codec",
    "codec = DataArray.codec(arr)",
    &[d::OUT_STRING],
    ArrayForm::Codec,
    CODEC_DOC,
    READ
);
