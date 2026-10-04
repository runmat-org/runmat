use super::super::documentation_support::*;
use crate::*;

const CUDA_SOURCE: &[BuiltinFilesystemEntry] = &[BuiltinFilesystemEntry::File {
    relative_path: "cuda_gateway.cu",
    content: BuiltinFixtureContent::Utf8("#include \"mex.h\"\nextern \"C\" void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { (void)nlhs; (void)plhs; (void)nrhs; (void)prhs; }\n"),
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("mexcuda"), slug: Some("mexcuda"),
    summary: "Build a native CUDA MEX module with the configured nvcc toolchain.",
    description: "`mexcuda(source, options...)` delegates to the shared MEX builder in CUDA mode. It requires at least one `.cu` translation unit and returns no value. CUDA compilation is currently supported only on native Linux and Windows targets with a compatible nvcc toolchain.",
    keywords: &["mexcuda", "mex", "CUDA", "nvcc", "GPU extension", "native build"],
    related: &["mex", "mexext"],
    sections: &[
        BuiltinDocumentationSection { heading: "CUDA build contract", paragraphs: &[
            "A `.cu` source is required even when C or C++ sources accompany it. Source paths, output naming, include/library flags, compiler assignments, verbosity, and the `-R2017b`/`-R2018a` API pins use the same parser as `mex`. Mixed Fortran/CUDA gateway sources are rejected by the adapter. The build records a target- and ABI-bound manifest for the resulting module.",
            "Scalar numbers and logical arguments are currently converted to text by RunMat's call boundary, like `mex`; this coercion is not MATLAB API parity. Unsupported targets are reported as UnsupportedTarget, invalid invocation options as InvalidBuildArguments, and compiler/SDK/output failures as BuildFailed with their build cause.",
        ] },
        BuiltinDocumentationSection { heading: "Execution limits", paragraphs: &[
            "CUDA compilation uses native process, filesystem, environment, and console effects. It is not itself a provider-resident array computation: the CUDA gateway's runtime behavior and ownership are validated separately when loaded. Building is unsupported in browser/Worker sessions and on native targets lacking the supported CUDA toolchain contract.",
            "The current compiler call blocks the caller. It cannot interrupt an in-flight compiler on invocation cancellation and does not enforce a complete-call host-memory budget.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "compile-cuda-gateway", title: "Build a CUDA gateway with the R2018a API", program: "mexcuda('-R2018a', '-output', 'cuda_gateway', 'cuda_gateway.cu');\nmodule = strcat('cuda_gateway.', mexext());\nassert(isfile(module));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::NativeFilesystem, fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture { id: BuiltinExampleFixtureId { local_name: "mex-cuda-gateway" }, root: BuiltinFilesystemRoot::IsolatedWorkspace, entries: CUDA_SOURCE }), requirements: CUDA_COMPILER, verification: BuiltinExampleVerification::Assertions { source: "assert(isfile(module));" } },
        BuiltinExample { id: "requires-source", title: "Require a source argument", program: "mexcuda();", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Native, fixture: BuiltinExampleFixture::None, requirements: NATIVE_TARGET, verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:MEX:InvalidBuildArguments" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Is a CUDA-capable GPU needed just to build?", answer: "The build requires a supported native CUDA compiler/toolchain; executing the compiled gateway has separate accelerator and loader requirements." },
        BuiltinDocumentationFaq { question: "Can I mix a Fortran gateway with CUDA?", answer: "No. The shared MEX builder rejects mixed Fortran and CUDA gateway sources." },
    ],
    links: &[MEX_LINK, EXTENSIONS_LINK, BuiltinDocumentationLink { label: "mex", target: BuiltinDocumentationLinkTarget::Builtin("mex") }, BuiltinDocumentationLink { label: "mexext", target: BuiltinDocumentationLinkTarget::Builtin("mexext") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: IMPLEMENTATION, verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CUDA source preflight", location: "builtins::interop::mex::tests::mexcuda_requires_a_cuda_translation_unit_on_supported_targets" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
