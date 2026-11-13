use super::*;

pub(in super::super) const LOAD: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("load"),
    slug: Some("load"),
    summary: "Load variables from a Level-5 MAT-file.",
    description: "`S = load(filename)` returns a scalar struct containing selected variables. A zero-output statement `load(filename)` instead assigns the selected variables into the caller workspace. Bare `load()` uses `matlab.mat` or the `RUNMAT_LOAD_DEFAULT_PATH` override.",
    keywords: &["load", "MAT-file", "workspace", "Level-5", "variables"],
    related: &["save", "matfile", "fileread", "loadobj"],
    sections: &[
        BuiltinDocumentationSection { heading: "Selection and outputs", paragraphs: &[
            "Use variable names after the filename, or `-regexp` with one or more patterns. `-mat` is accepted; ASCII/text loading and HDF5-backed v7.3 MAT-files are not supported. A missing requested variable or invalid regex raises a typed error. The zero-output form overwrites matching workspace bindings; capturing one output does not mutate the workspace. More than one output rejects before reading the file.",
            "A filename without an extension gains `.mat`. The default path is read from `RUNMAT_LOAD_DEFAULT_PATH` only for the bare/default form. Relative paths use the provider and working folder captured for this invocation.",
        ] },
        BuiltinDocumentationSection { heading: "Values and layout", paragraphs: &[
            "The Level-5 reader restores real and complex single/double values; all eight signed/unsigned integer classes with exact storage; logical, character, string, cell, structure, and sparse arrays; and RunMat typed object records. Shapes and column-major order are retained, including empty structure schemas. Registered static `Class.loadobj` callbacks restore saved objects.",
            "Decoded values are host-resident. Saving a GPU value materializes it through its owner; loading that file does not create a GPU allocation. Use `gpuArray` explicitly if a device result is required.",
        ] },
        BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
            "Input bytes are obtained from the captured filesystem provider. Large or compressed files, restoration callbacks, and workspace publication can use substantially more host memory than the file size. Check available memory before loading large or untrusted MAT files.",
        ] },
    ],
    examples: &[
        example("workspace", "Load into the caller workspace", "weights = [0.5, 1.0, 1.5];\nsave('results.mat', 'weights');\nclear weights;\nload('results.mat');\ndisp(norm(weights))", None, "assert(isequal(weights, [0.5 1.0 1.5]));"),
        BuiltinExample { id: "names", title: "Select variables by name", program: "time = 0:0.25:1;\nstate = [0, 0.5, 0.8, 0.6, 0.3];\nsave('sim_state.mat', 'time', 'state');\nclear time state;\nload('sim_state.mat', 'state', 'time');\nplot(time, state)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements { runtime: &[BuiltinRuntimeCapability::PlotCore], ..BuiltinExampleRequirements::NONE }, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(time, 0:0.25:1)); assert(isequal(state, [0 0.5 0.8 0.6 0.3]));" } },
        example("regexp", "Select variables with a regular expression", "layer_1 = ones(4, 4);\nlayer_2 = ones(8, 8);\nsave('checkpoint.mat', 'layer_1', 'layer_2');\nS = load('checkpoint.mat', '-regexp', '^layer_\\d+$')", None, "assert(isfield(S, 'layer_1')); assert(isfield(S, 'layer_2')); assert(isequal(size(S.layer_2), [8 8]));"),
        example("struct", "Capture a struct without workspace mutation", "x = 3.14;\ny = 2.72;\nsave('snapshot.mat', 'x', 'y');\nS = load('snapshot.mat');\ndisp(fieldnames(S))", None, "assert(S.x == 3.14); assert(S.y == 2.72);"),
        example("combined", "Combine names and regex filters", "config = struct('lr', 0.01);\nweights_conv = ones(3, 3);\nweights_fc = ones(4, 2);\nsave('model.mat', 'config', 'weights_conv', 'weights_fc');\nmodel = load('model.mat', 'config', '-regexp', '^weights_(conv|fc)')", None, "assert(isfield(model, 'config')); assert(isfield(model, 'weights_conv')); assert(isfield(model, 'weights_fc'));"),
        example("default", "Use a custom default MAT path", "x = 0;\nsave('autosave.mat', 'x');\nprevious = getenv('RUNMAT_LOAD_DEFAULT_PATH');\nsetenv('RUNMAT_LOAD_DEFAULT_PATH', 'autosave.mat');\nS = load();\nif isempty(previous), unsetenv('RUNMAT_LOAD_DEFAULT_PATH'); else, setenv('RUNMAT_LOAD_DEFAULT_PATH', previous); end", None, "assert(S.x == 0);"),
        example("text", "Load character data", "labels = 'hello';\nsave('strings.mat', 'labels');\nvalues = load('strings.mat', 'labels');\ndisp(values.labels)", Some("hello"), "assert(strcmp(values.labels, 'hello'));"),
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `load` support ASCII text files?", answer: "No. RunMat currently reads Level-5 MAT-files only; `load(filename, '-ascii')` and automatic ASCII loading are a coverage gap. Use text I/O utilities such as `readmatrix` or `readtable`." },
        BuiltinDocumentationFaq { question: "How are structures handled?", answer: "Scalar and array structures retain field schema, dimensions, and column-major element order, including empty schemas. Nested cells, logical, sparse, and typed object values are supported. A struct's field names never grant it object identity." },
        BuiltinDocumentationFaq { question: "Does `load` call `loadobj` for saved objects?", answer: "Yes. RunMat typed object records dispatch registered static `Class.loadobj` methods during restoration when available." },
        BuiltinDocumentationFaq { question: "Will `load` overwrite existing variables?", answer: "A zero-output call assigns matching names in the caller workspace. `S = load(...)` returns a struct without those assignments." },
        BuiltinDocumentationFaq { question: "What if a requested variable is missing?", answer: "The call raises a selection error naming the missing variable." },
        BuiltinDocumentationFaq { question: "Can I load into another workspace?", answer: "`load` targets the caller workspace in its statement form. Capture a struct and assign explicitly when a different scope is needed." },
        BuiltinDocumentationFaq { question: "How are GPU arrays handled?", answer: "Saved resident values are materialized to host data. Loaded values remain host-resident until an explicit device conversion or later acceleration." },
        BuiltinDocumentationFaq { question: "How do I inspect loaded names?", answer: "Capture the struct output and use `fieldnames` or `isfield`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "save", target: BuiltinDocumentationLinkTarget::Builtin("save") },
        BuiltinDocumentationLink { label: "who", target: BuiltinDocumentationLinkTarget::Builtin("who") },
        BuiltinDocumentationLink { label: "matfile", target: BuiltinDocumentationLinkTarget::Builtin("matfile") },
        BuiltinDocumentationLink { label: "fileread", target: BuiltinDocumentationLinkTarget::Builtin("fileread") },
        BuiltinDocumentationLink { label: "loadobj", target: BuiltinDocumentationLinkTarget::Builtin("loadobj") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "MAT load entry and selection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/load.rs") },
            BuiltinDocumentationLink { label: "MAT load controls and variable selection", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/load/controls.rs") },
            BuiltinDocumentationLink { label: "MAT load source read and value restoration", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/load/reader.rs") },
            BuiltinDocumentationLink { label: "MAT element decoder and typed values", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/load/decoder.rs") },
            BuiltinDocumentationLink { label: "MAT tag and payload validation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/load/decoder/tagged.rs") },
            BuiltinDocumentationLink { label: "Captured MAT filesystem authority", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/mat/authority.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "MAT load decoder and values", location: "builtins::io::mat::load::tests" }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
