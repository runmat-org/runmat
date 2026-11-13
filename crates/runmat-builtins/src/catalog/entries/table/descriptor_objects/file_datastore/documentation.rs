use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fileDatastore"),
    slug: Some("fileDatastore"),
    summary: "Create a file datastore descriptor object.",
    description: "`fileDatastore(files, 'ReadFcn', reader, Name, Value...)` creates a descriptor object for file-backed datastore workflows. It preserves file specifications and reader metadata without opening files or invoking the reader.",
    keywords: &["fileDatastore", "datastore", "file", "readfcn"],
    related: &["arrayDatastore", "parquetDatastore"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported construction",
            paragraphs: &[
                "Text file specifications may be a character row vector, string scalar or array, or cell array of character row vectors. Files retains the original specification; wildcard expansion, path existence checks, directory enumeration, and FileSet/DsFileSet interpretation are not performed.",
                "MATLAB-compatible construction requires ReadFcn to be a function handle or closure. The supplied callable and its captured values are retained without resolving its name, validating its invocation signature, or executing it. For example, a partial-file reader's state and completion protocol are not exercised during construction.",
                "Supported name/value options are ReadFcn, FileExtensions, IncludeSubfolders, and ReadMode. Names are case-insensitive. Duplicate options use the last value; each supplied control is validated. FileExtensions accepts text, string arrays, or cellstr and defaults to an empty 0-by-1 string array (no extension restriction). Its original representation is retained.",
                "IncludeSubfolders accepts a logical scalar or double scalar zero/one and is stored as a logical scalar, default false. ReadMode accepts 'file' (default) or 'partialfile', case-insensitively, and retains the supplied spelling.",
                "There is exactly one descriptor output. Calls without a retained output still validate and construct the descriptor; requests for more than one output reject. Missing files, unpaired options, unsupported names, invalid extension lists, and invalid controls report an argument error.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "RunMat metadata extensions",
            paragraphs: &[
                "RunMat extension mode also permits an omitted ReadFcn (stored as an empty string), a non-callable ReadFcn payload, arbitrary opaque Files metadata, and nested text cells. These forms store values only; an integer or string reader payload is not converted into an executable callback.",
                "In RunMat extension mode, IncludeSubfolders additionally accepts a native integer scalar (exact zero/nonzero), any finite double scalar, or the case-insensitive text aliases true/on/yes and false/off/no. Non-scalar, non-finite, single-precision, and integer-array controls reject.",
                "Datastore descriptors are host metadata, not GPU kernels or fused operations. Explicit resident inputs require the resident-input RunMat extension and gather through their owning provider. Retained numeric payloads preserve class, shape, complexity and exact native integer values; IncludeSubfolders alone is converted to logical. No arithmetic precision conversion is performed on Files or ReadFcn payloads.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Current limits",
            paragraphs: &[
                "This constructor provides metadata only. Generic file datastore read, readall, preview, hasdata, reset, partition, and streaming consumers are not implemented by it. Storing a valid ReadFcn does not enable file iteration or partial-file execution.",
                "ReadMode 'bytes', BlockSize, PreviewFcn, UniformRead, AlternateFileSystemRoots, CreateLocalCopy, and other unlisted options are not supported and reject. No filesystem service or callback execution capability is required just to construct this descriptor. The host constructor is available on native and WebAssembly targets; it has no distributed execution protocol.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "standalone-descriptor-contract",
        title: "Create file datastore descriptor",
        program: "ds = fileDatastore('*.txt', 'ReadFcn', @readlines)",
        display_output: Some("ds = fileDatastore object"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(ds, 'fileDatastore'));\nassert(strcmp(ds.Files, '*.txt'));\nassert(isa(ds.ReadFcn, 'function_handle'));\nassert(isempty(ds.FileExtensions));\nassert(isequal(ds.IncludeSubfolders, false));\nassert(strcmp(ds.ReadMode, 'file'));",
        },
    }],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "arrayDatastore", target: BuiltinDocumentationLinkTarget::Builtin("arrayDatastore") },
        BuiltinDocumentationLink { label: "parquetDatastore", target: BuiltinDocumentationLinkTarget::Builtin("parquetDatastore") },
        BuiltinDocumentationLink { label: "MATLAB fileDatastore reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.filedatastore.html") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/table/builtins/constructors.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/constructors.rs"),
        }, BuiltinDocumentationLink {
            label: "File datastore metadata construction and compatibility validation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/table/builtins/constructors/file_datastore.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::table::tests", location: "crates/runmat-runtime/src/builtins/table/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "File datastore typed contract and inference", location: "crates/runmat-builtins/src/catalog/entries/table/descriptor_objects/file_datastore/tests.rs" },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
