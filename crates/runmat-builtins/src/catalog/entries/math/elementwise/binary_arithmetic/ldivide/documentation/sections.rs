use crate::BuiltinDocumentationSection;

pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Operand direction and expansion",
        paragraphs: &[
            "The first input is the divisor and the second is the numerator. Equal dimensions align directly, and singleton dimensions expand when all non-singleton extents are compatible. Incompatible dimensions produce a size-mismatch error; compatible empty dimensions remain empty in the broadcasted shape.",
            "Logical inputs contribute zero or one, and character arrays contribute Unicode code points. String arrays and other nonnumeric containers are rejected. Mixed real and complex inputs return a complex quotient.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and numerical behavior",
        paragraphs: &[
            "Double floating-point inputs produce double output. If either floating-point input is single, the result uses single or complex-single storage. Because `ldivide(A, B)` computes `B ./ A`, zero behavior is determined by zero values in `A`.",
            "Matching fixed-width integer classes preserve that class. One integer operand may instead be paired with scalar double. Integer quotients round to nearest with half ties away from zero and saturate at the destination bounds. Complex integer arithmetic is rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Provider execution and fusion",
        paragraphs: &[
            "Provider hooks receive the numerator before the denominator. A common provider can execute matching resident pairs, real scalar forms, exact typed-integer forms, and supported singleton expansion. Returned handles are validated for owner, device, storage kind, class, precision, shape, and non-aliasing.",
            "Unsupported forms gather through their owning provider and execute the same host contract. Explicit device intent is restored to the selected provider or the call fails; automatic placement may safely return a host value. Compatible floating-point expressions remain eligible for element-wise fusion.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Output prototype extension",
        paragraphs: &[
            "RunMat mode accepts `ldivide(A, B, 'like', prototype)`. The prototype selects host or provider residency and can request complex output. MATLAB compatibility mode rejects this extension before execution.",
        ],
    },
];
