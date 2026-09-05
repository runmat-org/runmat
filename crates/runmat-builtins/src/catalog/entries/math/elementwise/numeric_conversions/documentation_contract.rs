pub(super) use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

macro_rules! define_integer_conversion_documentation {
    ($name:literal, $summary:literal, $description:literal, $range:literal, $saturation_program:literal, $saturation_output:literal, $saturation_assertion:literal, $gpu_input:literal) => {
        const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
            implementation: &[BuiltinDocumentationLink {
                label: "Runtime implementation",
                target: BuiltinDocumentationLinkTarget::Source(concat!(
                    "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/integer_conversions/",
                    $name,
                    "/mod.rs"
                )),
            }],
            verification: &[
                BuiltinEvidenceReference {
                    kind: BuiltinEvidenceKind::UnitTest,
                    label: "Exact integer conversion matrix",
                    location: "builtins::math::elementwise::integer_conversions::tests::host",
                },
                BuiltinEvidenceReference {
                    kind: BuiltinEvidenceKind::ProviderTest,
                    label: "Resident typed conversion matrix",
                    location: "builtins::math::elementwise::integer_conversions::tests::provider",
                },
            ],
            notes: &[],
        };

        const SECTIONS: &[BuiltinDocumentationSection] = &[
            BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                concat!("`Y = ", $name, "(X)` converts supported values to native `", $name, "` storage and preserves every array dimension. The class range is ", $range, "."),
                "Finite fractional values round to the nearest integer with halfway cases away from zero, then saturate to the target range. Negative values saturate to zero for unsigned classes. Positive and negative infinity saturate at the corresponding boundary, and NaN becomes zero.",
                "Every fixed-width integer input converts directly from authoritative native storage without passing through binary64. Logical values become zero or one, characters become Unicode code points, and real sparse input retains CSC structure with native integer stored values.",
                "Real and complex numeric, logical, character, symbolic-constant, sparse, and supported gpuArray input is accepted. Complex values convert both components and remain complex even when the converted imaginary component is zero. String, cell, struct, object, handle, and nonconstant symbolic input returns a typed conversion error.",
            ] },
            BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
                concat!("Supported providers convert real resident input directly to native `", $name, "` storage and retain the owning provider and shape."),
                "When a provider lacks the typed conversion hook, RunMat gathers through the exact owner, applies the same saturating conversion to authoritative host storage, and restores a resident result when representable. It never labels a floating buffer as an integer class.",
            ] },
        ];

        const EXAMPLES: &[BuiltinExample] = &[
            BuiltinExample { id: "saturation", title: "Round and saturate values at the class boundaries", program: $saturation_program, display_output: Some($saturation_output), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: $saturation_assertion } },
            BuiltinExample { id: "shape-and-class", title: "Preserve matrix shape in native integer storage", program: concat!("A = [1 2; 3 4];\nY = ", $name, "(A)"), display_output: Some(concat!("Y is a 2-by-2 ", $name, " matrix")), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isa(Y, \"", $name, "\"));\nassert(isequal(size(Y), [2 2]));\nassert(isequal(Y, ", $name, "([1 2; 3 4])));") } },
            BuiltinExample { id: "gpu-array", title: "Convert provider-resident values to native integer storage", program: concat!("G = gpuArray(", $gpu_input, ");\nY = ", $name, "(G);\nhost = gather(Y)"), display_output: Some(concat!("host is a 2-by-2 ", $name, " matrix")), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: concat!("assert(isa(host, \"", $name, "\"));\nassert(isequal(host, ", $name, "(", $gpu_input, ")));\nassert(isequal(size(host), [2 2]));") } },
        ];

        const FAQS: &[BuiltinDocumentationFaq] = &[
            BuiltinDocumentationFaq { question: concat!("Does `", $name, "` preserve array shape?"), answer: concat!("Yes. Every dimension is retained while the stored numeric class becomes `", $name, "`.") },
            BuiltinDocumentationFaq { question: "How are out-of-range and fractional values handled?", answer: "Values round to the nearest integer with halfway cases away from zero and then saturate at the target boundary." },
            BuiltinDocumentationFaq { question: "Does conversion pass integers through floating point?", answer: "No. Fixed-width integer input converts directly from its authoritative native storage, including exact 64-bit values." },
            BuiltinDocumentationFaq { question: "What happens to complex input?", answer: concat!("Both components convert independently to `", $name, "`, and the result retains complex storage even when one converted component is zero.") },
            BuiltinDocumentationFaq { question: "What happens to sparse input?", answer: concat!("Real sparse input retains CSC structure and converts its stored values to native `", $name, "` storage.") },
            BuiltinDocumentationFaq { question: "Will gpuArray input remain resident?", answer: "Yes when the owning provider supports native integer storage or the typed owner-aware fallback can restore the result." },
        ];

        pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($name),
            slug: Some($name),
            summary: $summary,
            description: $description,
            keywords: &[$name, "integer", "cast", "saturating conversion", "gpuArray", "native storage"],
            related: &["double", "single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "intmin", "intmax", "gather", "gpuArray"],
            sections: SECTIONS,
            examples: EXAMPLES,
            example_exemption: None,
            faqs: FAQS,
            links: &[],
            media: &[],
            evidence: EVIDENCE,
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Stable),
        };
    };
}

pub(super) use define_integer_conversion_documentation;
