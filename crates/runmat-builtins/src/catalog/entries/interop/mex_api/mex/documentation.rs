use super::super::documentation_support::*;
use crate::*;

const C_SOURCE: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
    relative_path: "gateway.c",
    content: BuiltinFixtureContent::Utf8("#include \"mex.h\"\nvoid mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { (void)nlhs; (void)plhs; (void)nrhs; (void)prhs; }\n"),
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mex"), slug: Some("mex"),
    summary: "Build a native C, C++, or Fortran MEX module for RunMat.",
    description: "`mex(source, options...)` compiles supported native translation units and publishes a RunMat-loadable module and deterministic artifact manifest. It is a build command with no returned value. It does not call the resulting gateway.",
    keywords: &["mex", "compile", "C MEX", "native extension", "build", "R2018a", "R2017b"],
    related: &["mexext", "mexcuda"],
    sections: &[
        BuiltinDocumentationSection { heading: "Sources and options", paragraphs: &[
            "Pass at least one C, C++, or Fortran source path. Relative paths and output directories resolve from the current process working directory. `-output` and `-outdir` select the artifact; `-I`, `-D`, `-L`, `-l`, selected compiler assignments, and `-v`/`-verbose`/`-silent` are parsed by the shared native MEX builder. Unsupported assignments and conflicting ABI switches raise InvalidBuildArguments; compiler and filesystem failures retain their underlying build cause.",
            "`-R2017b` and `-R2018a` select the compatibility API. `-compatibleArrayDims` and `-largeArrayDims` are additional supported API selectors, with builder-owned conflict rules and distinct dimension contracts. The selected ABI is recorded in the artifact manifest; RunMat checks the manifest before loading. The bundled `mex.h` and MEX SDK are prepared by the native build adapter, not by this builtin.",
            "The current RunMat call boundary converts scalar numeric and logical arguments to text as well as accepting scalar strings and one-row character vectors. This is existing RunMat coercion, not a claim of MATLAB argument parity. Other values fail before invoking the compiler.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and ownership", paragraphs: &[
            "Compilation runs on a supported native target with a compatible C-family or Fortran toolchain and writable output directory. The command may read environment/toolchain configuration, source and SDK files, write modules/manifests, and print progress. It is not a GPU kernel, fusion candidate, or distributable partition operation. Browser/Worker execution cannot create a native MEX module.",
            "Building a MEX module does not itself transfer or copy MATLAB array payloads. At invocation time the native MEX adapter validates its manifest, ABI and callback contract; zero-copy views remain owner-bound and copy-on-write mutation rules apply as described in the MEX execution guide. Build success alone does not prove callback isolation or later module execution.",
            "The current compiler call blocks the caller and has no invocation cancellation or finite host-resource admission. A cancelled caller cannot assume an in-flight compiler was stopped; source, SDK, compiler, and output allocations are not governed by a complete-call memory budget.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "compile-c-gateway", title: "Build a C gateway with the R2018a API", program: "mex('-R2018a', '-output', 'gateway', 'gateway.c');\nmodule = strcat('gateway.', mexext());\nassert(isfile(module));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "mex-c-gateway" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: C_SOURCE }), requirements: NATIVE_COMPILER, verification: BuiltinExampleVerification::Assertions { source: "assert(isfile(module));" } },
        BuiltinExample { id: "missing-source", title: "Reject a missing source argument", program: "mex();", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Native, fixture: BuiltinExampleFixture::None, requirements: NATIVE_TARGET, verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:MEX:InvalidBuildArguments" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does `mex` run the new module?", answer: "No. It compiles and publishes the module and artifact manifest; calling the resulting function is a separate operation." },
        BuiltinDocumentationFaq { question: "Which API switch should I use?", answer: "Use -R2018a for interleaved-complex C Matrix API compatibility or -R2017b for the separate-complex compatibility path. The chosen API is pinned in the artifact manifest." },
        BuiltinDocumentationFaq { question: "Can it compile on a browser Worker?", answer: "No. The current builder and native module loader require a supported native target and toolchain." },
    ],
    links: &[MEX_LINK, EXTENSIONS_LINK, BuiltinDocumentationLink { label: "mexext", target: BuiltinDocumentationLinkTarget::Builtin("mexext") }, BuiltinDocumentationLink { label: "mexcuda", target: BuiltinDocumentationLinkTarget::Builtin("mexcuda") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Native MEX build", location: "builtins::interop::mex::tests::mex_builds_source_with_the_shared_adapter_service" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
