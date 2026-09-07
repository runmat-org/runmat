use crate::BuiltinDocumentationSection;

pub(super) const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Operands and expansion",
        paragraphs: &[
            "Equal dimensions align directly, and singleton dimensions expand when every non-singleton extent is compatible. Incompatible dimensions produce a size-mismatch error. Compatible empty dimensions remain empty in the broadcasted result.",
            "Logical inputs contribute zero or one, and character arrays contribute Unicode code points. Scalar symbolic inputs and compatible symbolic arrays construct symbolic power expressions.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Real, complex, and integer domains",
        paragraphs: &[
            "Negative real bases with noninteger exponents produce principal complex results instead of NaN. Complex bases or exponents follow principal complex exponentiation. Double data produces double or complex-double output; participating single data produces single or complex-single output.",
            "Fixed-width integer bases preserve their class and require nonnegative integer-valued exponents. Exponentiation is exact within the class and saturates at its bounds. Scalar-double compatibility uses exact wide-integer admission, and unsupported complex-integer combinations are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Provider execution and fusion",
        paragraphs: &[
            "Matching provider-resident floating-point arrays use the provider's element-wise power hook. Supported scalar counterparts are materialized in compatible device storage, and compatible singleton expansion can use provider-side replication before exponentiation.",
            "Integer and complex forms, invalid integer exponents, unsupported shapes, and providers without the required operation gather authoritative storage and use the host contract. Compatible real floating-point expressions can be fused with adjacent element-wise work. Every accepted resident result is distinct from its inputs and retains the expected owner, device, class, storage, precision, and shape.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Output prototype extension",
        paragraphs: &[
            "RunMat mode accepts `power(A, B, 'like', prototype)`. The prototype selects host or provider residency and can promote a real result to complex storage. MATLAB compatibility mode rejects this extension before execution.",
        ],
    },
];
