use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

macro_rules! define_integer_documentation {
    (
        $module:ident,
        $export:ident,
        $name:literal,
        $summary:literal,
        $description:literal,
        $range:literal,
        $saturation_program:literal,
        $saturation_output:literal,
        $saturation_assertion:literal,
        $gpu_input:literal,
        $implementation:literal
    ) => {
        mod $module {
            use super::*;

            const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
                implementation: &[BuiltinDocumentationLink {
                    label: "Runtime implementation",
                    target: BuiltinDocumentationLinkTarget::Source($implementation),
                }],
                verification: &[
                    BuiltinEvidenceReference {
                        kind: BuiltinEvidenceKind::UnitTest,
                        label: "Exact integer conversion tests",
                        location: "builtins::math::elementwise::integer_cast_builtins::tests",
                    },
                    BuiltinEvidenceReference {
                        kind: BuiltinEvidenceKind::ProviderTest,
                        label: "Resident typed conversion tests",
                        location: "integer_conformance::gpu_conversion_preserves_native_integer_storage",
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

            pub(in super::super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
                authority: BuiltinDocumentationAuthority::Catalog,
                title: Some($name), slug: Some($name),
                summary: $summary,
                description: $description,
                keywords: &[$name, "integer", "cast", "saturating conversion", "gpuArray", "native storage"],
                related: &["double", "single", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "intmin", "intmax", "gather", "gpuArray"],
                sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
                links: &[], media: &[], evidence: EVIDENCE, introduced: None,
                status: Some(BuiltinDocumentationStatus::Stable),
            };
        }

        pub(super) use $module::DOCUMENTATION as $export;
    };
}

define_integer_documentation!(int8, INT8_DOCUMENTATION, "int8", "Convert supported values to signed 8-bit integer storage.", "`int8(X)` performs shape-preserving, rounded, saturating conversion to native signed 8-bit storage.", "-128 through 127", "values = int8([-Inf -1.5 0 1.5 Inf])", "values = int8([-128 -2 0 2 127])", "assert(isa(values, \"int8\"));\nassert(isequal(values, [intmin(\"int8\") int8(-2) int8(0) int8(2) intmax(\"int8\")]));", "[-2 0; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/integer_cast_builtins.rs");
define_integer_documentation!(int16, INT16_DOCUMENTATION, "int16", "Convert supported values to signed 16-bit integer storage.", "`int16(X)` performs shape-preserving, rounded, saturating conversion to native signed 16-bit storage.", "-32,768 through 32,767", "values = int16([-Inf -1.5 0 1.5 Inf])", "values = int16([-32768 -2 0 2 32767])", "assert(isa(values, \"int16\"));\nassert(isequal(values, [intmin(\"int16\") int16(-2) int16(0) int16(2) intmax(\"int16\")]));", "[-2 0; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/integer_cast_builtins.rs");
define_integer_documentation!(int32, INT32_DOCUMENTATION, "int32", "Convert supported values to signed 32-bit integer storage.", "`int32(X)` performs shape-preserving, rounded, saturating conversion to native signed 32-bit storage.", "-2,147,483,648 through 2,147,483,647", "values = int32([-Inf -1.5 0 1.5 Inf])", "values = int32([-2147483648 -2 0 2 2147483647])", "assert(isa(values, \"int32\"));\nassert(isequal(values, [intmin(\"int32\") int32(-2) int32(0) int32(2) intmax(\"int32\")]));", "[-2 0; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/int32.rs");
define_integer_documentation!(int64, INT64_DOCUMENTATION, "int64", "Convert supported values to signed 64-bit integer storage.", "`int64(X)` performs shape-preserving, rounded, saturating conversion to native signed 64-bit storage.", "-9,223,372,036,854,775,808 through 9,223,372,036,854,775,807", "values = int64([-Inf -1.5 0 1.5 Inf])", "values contains int64 minimum, -2, 0, 2, and int64 maximum", "assert(isa(values, \"int64\"));\nassert(isequal(values, [intmin(\"int64\") int64(-2) int64(0) int64(2) intmax(\"int64\")]));", "[-2 0; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/integer_cast_builtins.rs");
define_integer_documentation!(uint8, UINT8_DOCUMENTATION, "uint8", "Convert supported values to unsigned 8-bit integer storage.", "`uint8(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 8-bit storage.", "0 through 255", "values = uint8([-Inf 0 3.5 Inf])", "values = uint8([0 0 4 255])", "assert(isa(values, \"uint8\"));\nassert(isequal(values, [uint8(0) uint8(0) uint8(4) intmax(\"uint8\")]));", "[0 2; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/uint8.rs");
define_integer_documentation!(uint16, UINT16_DOCUMENTATION, "uint16", "Convert supported values to unsigned 16-bit integer storage.", "`uint16(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 16-bit storage.", "0 through 65,535", "values = uint16([-Inf 0 3.5 Inf])", "values = uint16([0 0 4 65535])", "assert(isa(values, \"uint16\"));\nassert(isequal(values, [uint16(0) uint16(0) uint16(4) intmax(\"uint16\")]));", "[0 2; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/uint16.rs");
define_integer_documentation!(uint32, UINT32_DOCUMENTATION, "uint32", "Convert supported values to unsigned 32-bit integer storage.", "`uint32(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 32-bit storage.", "0 through 4,294,967,295", "values = uint32([-Inf 0 3.5 Inf])", "values = uint32([0 0 4 4294967295])", "assert(isa(values, \"uint32\"));\nassert(isequal(values, [uint32(0) uint32(0) uint32(4) intmax(\"uint32\")]));", "[0 2; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/uint32.rs");
define_integer_documentation!(uint64, UINT64_DOCUMENTATION, "uint64", "Convert supported values to unsigned 64-bit integer storage.", "`uint64(X)` performs shape-preserving, rounded, saturating conversion to native unsigned 64-bit storage.", "0 through 18,446,744,073,709,551,615", "values = uint64([-Inf 0 3.5 Inf])", "values contains 0, 0, 4, and uint64 maximum", "assert(isa(values, \"uint64\"));\nassert(isequal(values, [uint64(0) uint64(0) uint64(4) intmax(\"uint64\")]));", "[0 2; 1 3]", "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/integer_cast_builtins.rs");
