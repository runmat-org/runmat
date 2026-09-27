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
            "Finite negative real bases with finite noninteger exponents produce principal complex results instead of NaN. The result keeps complex storage even when its imaginary component is tiny or underflows to zero. Complex bases or exponents follow principal complex exponentiation. Double data produces double or complex-double output; participating single data produces single or complex-single output.",
            "Real NaN inputs propagate NaN, including power(1, NaN) and power(NaN, 0). Real infinity and signed-zero pairs retain floating-power behavior; for example, power(-0, -3) is negative infinity and power(-Inf, 0.5) is positive infinity.",
            "Fixed-width integer bases preserve their class and require nonnegative integer-valued exponents. Exponentiation is exact within the class and saturates at its bounds. Scalar-double compatibility uses exact wide-integer admission, and unsupported complex-integer combinations are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Provider execution and fusion",
        paragraphs: &[
            "Provider-resident real floating arrays are classified at their actual broadcast positions before a domain-preserving power kernel selects real or interleaved complex output. The WGPU classifier reduces device flags and reads back one scalar domain flag; it does not download the input arrays. Its allocation, dispatch, synchronization, and readback costs participate in placement estimates.",
            "Integer and already-complex operands, unsupported representations, and providers without the required domain operation gather authoritative storage and use the host contract. A legacy real-only provider hook remains usable for scalar forms whose original scalar proves a real domain without changing NaN behavior. Every accepted resident result is distinct from its inputs and retains the expected owner, device, class, storage, precision, and shape. Fusion also requires a valid result-domain contract; real input storage alone is insufficient.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Output prototype extension",
        paragraphs: &[
            "RunMat mode accepts `power(A, B, 'like', prototype)`. The prototype selects host or provider residency and can promote a real result to complex storage. MATLAB compatibility mode rejects this extension before execution.",
        ],
    },
];
