use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("interp1q"),
    slug: Some("interp1q"),
    summary: "Legacy quick one-dimensional linear interpolation.",
    description: "`interp1q(X,V,Xq)` linearly interpolates values at query locations. It is a legacy entry point; use `interp1` for new code.",
    keywords: &["interp1q", "interpolation", "linear", "legacy"],
    related: &["interp1"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Syntax and input geometry",
            paragraphs: &[
                "`Vq = interp1q(X,V,Xq)` takes exactly three inputs. X is a finite strictly increasing vector with at least two sample points. Vector V follows X; matrix or N-D V has one sample series per trailing component coordinate. Only linear interpolation is available, and out-of-range query points return NaN.",
                "Complex V values interpolate real and imaginary components separately. Complex X and Xq are admitted only when their imaginary components are zero. Vector V preserves query shape; matrix V returns `numel(Xq)` by its trailing series dimensions.",
                "MATLAB's public interp1q documentation specifies column-vector X and Xq and marks the function not recommended. RunMat retains its prior row-vector and array-query behavior as a RunMat compatibility form; this is not a claim that those shapes match MATLAB's documented entry point.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Classes and execution",
            paragraphs: &[
                "The current CPU implementation returns double real or complex values. Typed integer X, V, and Xq require exact conversion to binary64 and are RunMat-mode extensions. Logical inputs and resident gpuArray inputs are independently gated RunMat-mode extensions.",
                "Resident inputs are gathered to the CPU linear interpolation implementation, including explicitly device-resident values. There is no interp1q GPU interpolation kernel or resident-output promise. The call is a fusion boundary.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "linear-interpolation",
            title: "Interpolate linearly",
            program: "vq = interp1q([1 2 3], [10 20 40], [1.5 2.5])",
            display_output: Some("vq = [15 30]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(vq)==2); assert(abs(vq(1)-15)<1e-12); assert(abs(vq(2)-30)<1e-12); assert(strcmp(class(vq),'double'));",
            },
        },
        BuiltinExample {
            id: "outside-sample-range",
            title: "Return NaN outside the sample range",
            program: "vq = interp1q([1 2], [10 20], 3)",
            display_output: Some("vq = NaN"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isnan(vq)); assert(strcmp(class(vq),'double'));",
            },
        },
        BuiltinExample {
            id: "complex-values",
            title: "Interpolate complex sample values",
            program: "vq = interp1q([1 2 3], [10+1i 20+3i 40+7i], [1.5 2.5])",
            display_output: Some("vq = [15+2i 30+5i]"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(vq)==2); assert(abs(vq(1)-(15+2i))<1e-12); assert(abs(vq(2)-(30+5i))<1e-12);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "interp1", target: BuiltinDocumentationLinkTarget::Builtin("interp1") },
        BuiltinDocumentationLink { label: "MATLAB interp1q reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/interp1q.html") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "interp1q runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/interpolation/interp1q.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "interp1q real, complex, and diagnostics", location: "builtins::math::interpolation::interp1q::tests" }],
        notes: &["The legacy sidecar's feature requirement is null, so this catalog entry declares no feature dependency. GPU precision information records accepted input types and the documented gather boundary, not a resident kernel."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
