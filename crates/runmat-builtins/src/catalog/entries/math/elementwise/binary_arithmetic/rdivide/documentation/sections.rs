use crate::BuiltinDocumentationSection;

pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Operands and expansion",
        paragraphs: &[
            "Equal dimensions align directly. A singleton dimension expands to the corresponding extent of the other operand. Incompatible non-singleton dimensions produce a size-mismatch error. If a compatible output dimension is zero, the result retains the broadcasted empty shape.",
            "Logical inputs contribute zero or one, and character arrays contribute Unicode code points. String arrays and other nonnumeric containers are rejected. Mixed real and complex inputs return a complex quotient.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and numerical behavior",
        paragraphs: &[
            "Double floating-point inputs produce double output. If either floating-point input is single, the result uses single or complex-single storage. Floating-point zero divisors follow IEEE-754 signed-infinity and NaN behavior; complex operands use the complex quotient.",
            "Matching fixed-width integer classes preserve that class. One integer operand may instead be paired with scalar double. Integer quotients round to nearest with half ties away from zero and saturate at the destination class bounds. Wide integer and scalar-double forms use the exact compatibility path rather than first rounding the integer through binary64.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Provider execution and fusion",
        paragraphs: &[
            "A common provider can execute matching resident pairs, real scalar forms, and supported singleton expansion. Every returned handle is checked for provider owner, device, storage kind, class, precision, shape, and non-aliasing before it is accepted.",
            "When a provider cannot satisfy the typed operation, RunMat gathers authoritative storage and evaluates the same host contract. Explicit device intent is restored to the owning provider or the call returns an error; automatically placed values may remain on the host. Compatible floating-point expressions can also be fused with adjacent element-wise work.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Output prototype extension",
        paragraphs: &[
            "RunMat mode accepts `rdivide(A, B, 'like', prototype)`. A host prototype requests host output, a provider-resident prototype requests output on that provider, and a complex prototype promotes a real quotient to complex storage. MATLAB compatibility mode rejects this extension before execution.",
        ],
    },
];
