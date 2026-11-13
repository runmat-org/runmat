use crate::*;

pub(super) const WHICH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some(r####"which"####),
    slug: Some(r####"which"####),
    summary: r####"Resolve which variable, builtin, file, or class matches a given name."####,
    description: r####"`which name` observes the current dynamic-call name environment without invoking a function or loading a native module. The runtime's workspace, semantic-function, builtin, native-module, and search-path owners supply the ordered candidates; `which` formats them."####,
    keywords: &[
        r####"which"####,
        r####"search path"####,
        r####"builtin lookup"####,
        r####"script path"####,
        r####"variable shadowing"####,
    ],
    related: &[],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior and placement",
            paragraphs: &[
                r####"Names can be supplied as character vectors or string scalars. Calling `which` with no name raises `which: not enough input arguments`."####,
                r####"`which name` without options returns the first candidate observed for an ordinary dynamic-name call. Workspace variables precede function candidates. Registered constructors can precede native, semantic, and source fallbacks; an exact executable builtin binding wins a same-name constructor collision. Otherwise the runtime observes native modules, semantic functions (including selected local/private bindings), `.m` source, and builtin bindings in call-route order. Presentation-only class folders and other files follow executable candidates."####,
                r####"`which(name, "-all")` (or `which("-all", name)`) returns a cell array of the selected candidate followed by ordered shadowed candidates, without duplicate displayed entries. When nothing matches, even `-all` returns the character row `'<name>' not found.` Ordinary lookup stops at the first match; `-all` continues through later candidates."####,
                r####"`which(..., "-builtin")` restricts the search to builtin functions. `"-var"` restricts to workspace variables, and `"-file"` restricts the search to files, classes, and folders."####,
                r####"Package-qualified names like `pkg.func` map to `+pkg` folders in the current path search. Registered class constructors and public static methods are observed where their dynamic-call routes apply; class candidates also include `@ClassName` folders and `.m` files containing `classdef`. An unqualified instance-method call can depend on receiver arguments, so a bare name alone cannot identify every method dispatch."####,
                r####"Relative paths are resolved against the current working directory. Absolute paths and paths beginning with `~` or drive letters are honoured directly."####,
                r####"`which` parses text and observes runtime/path metadata on the host. A GPU tensor has a numeric descriptor, not a character or string representation, so it is rejected as a name or option before a provider download. Results are host-resident character arrays or cell arrays."####,
                r####"Named `feval` has a distinct semantic-before-builtin route. `which` reports the ordinary dynamic-name route; it does not claim that the same ordering applies to every call form."####,
            ],
        },
        BuiltinDocumentationSection {
            heading: "Examples and displayed results",
            paragraphs: &[
                r####"Finding a built-in function's implementation
which("sin")
Expected display: built-in (RunMat builtin: sin)"####,
                r####"Checking if a workspace variable shadows a builtin
answer = 42;
which("answer")
Expected display: 'answer' is a variable."####,
                r####"Listing all matches on the path (illustrative when both implementations exist; order follows the active dynamic-call environment)
which("sum", "-all")
Expected display: {
    [1,1] = built-in (RunMat builtin: sum)
    [2,1] = //runmat/stdlib/sum.m
}"####,
                r####"Locating a script or function file
which("helpers/process_data")
Expected display: //projects/runmat/helpers/process_data.m"####,
                r####"Restricting the search to variables
which("-var", "velocity")
Expected display: 'velocity' is a variable."####,
                r####"Restricting the search to files
which("fft", "-file")
Expected display: //runmat/overrides/fft.m"####,
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "typed-basic",
        title: "resolve a builtin",
        program: "p = which('sin')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(p, 'built-in (RunMat builtin: sin)'));",
        },
    },
    BuiltinExample {
        id: "typed-variable",
        title: "observe a workspace variable",
        program: "resolution_which_example_value = 42; p = which('resolution_which_example_value')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(p, \"'resolution_which_example_value' is a variable.\"));",
        },
    },
    BuiltinExample {
        id: "typed-all-no-match",
        title: "report no match with -all",
        program: "p = which('__runmat_which_documentation_no_match__', '-all')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(p, \"'__runmat_which_documentation_no_match__' not found.\"));",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: r####"What happens when nothing is found?"####,
            answer: r####"`which` returns the character vector `'<name>' not found.`, including when `-all` is requested."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Are method lookups supported?"####,
            answer: r####"Registered constructors and public static methods are included when a dynamic-name call can select them. `@Class` folders and `.m` files containing `classdef` are reported as file candidates. Unqualified instance methods still require a receiver to determine dispatch."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does `which` canonicalise paths?"####,
            answer: r####"Yes. RunMat reports canonical absolute paths where possible; when canonicalisation fails, the original path is returned."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Can I combine `-all` with other options?"####,
            answer: r####"Yes. For example, `which("plot", "-all", "-file")` lists every file-based implementation without reporting builtins or variables."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does the search respect `RUNMAT_PATH` / `MATLABPATH`?"####,
            answer: r####"Yes. The directory list mirrors the logic used by other REPL filesystem builtins."####,
        },
        BuiltinDocumentationFaq {
            question: r####"What about Simulink models or Java classes?"####,
            answer: r####"The general path search does not append `.slx`, `.mdl`, or `.class` extensions automatically. An explicitly spelled path may match if the VFS exposes that exact file; no Simulink or Java class interpretation is implied."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Are duplicate results filtered?"####,
            answer: r####"Yes. The first occurrence of each unique path is returned."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Is the lookup case sensitive?"####,
            answer: r####"Option flags are decoded case-insensitively. File and directory candidates use their constructed spelling and the active VFS, so case behavior depends on that filesystem; cross-platform case-insensitive lookup is not guaranteed."####,
        },
        BuiltinDocumentationFaq {
            question: r####"Does `which` gather GPU values?"####,
            answer: r####"No. GPU tensor descriptors are numeric rather than text, so a GPU tensor supplied as the name or an option is rejected before any provider download."####,
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: r####"exist"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./exist"####),
        },
        BuiltinDocumentationLink {
            label: r####"ls"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./ls"####),
        },
        BuiltinDocumentationLink {
            label: r####"dir"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./dir"####),
        },
        BuiltinDocumentationLink {
            label: r####"copyfile"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./copyfile"####),
        },
        BuiltinDocumentationLink {
            label: r####"class"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./class"####),
        },
        BuiltinDocumentationLink {
            label: r####"isa"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./isa"####),
        },
        BuiltinDocumentationLink {
            label: r####"ischar"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./ischar"####),
        },
        BuiltinDocumentationLink {
            label: r####"isstring"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./isstring"####),
        },
        BuiltinDocumentationLink {
            label: r####"who"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./who"####),
        },
        BuiltinDocumentationLink {
            label: r####"whos"####,
            target: BuiltinDocumentationLinkTarget::Documentation(r####"./whos"####),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: r####"`crates/runmat-runtime/src/builtins/introspection/which.rs`"####,
            target: BuiltinDocumentationLinkTarget::Source(
                r####"https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/introspection/which.rs"####,
            ),
        },
        BuiltinDocumentationLink { label: "Catalog contract", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/resolution/contract.rs") },
        BuiltinDocumentationLink { label: "Catalog inference", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-builtins/src/catalog/entries/introspection/resolution/inference.rs") }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime which tests",
                location: r####"builtins::introspection::which::tests"####,
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Workspace observation test",
                location: r####"builtins::introspection::which::tests::which_variable_search_respects_workspace"####,
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
