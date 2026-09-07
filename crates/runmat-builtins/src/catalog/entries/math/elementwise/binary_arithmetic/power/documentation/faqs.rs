use super::super::super::documentation::faq;
use crate::BuiltinDocumentationFaq;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    faq(
        "Does `power` support implicit expansion?",
        "Yes. Singleton dimensions expand automatically, and incompatible non-singleton dimensions produce a size-mismatch error.",
    ),
    faq(
        "What numeric class does `power` return?",
        "Floating-point output is double unless single data participates. Results become complex when exponentiation leaves the real line. Supported integer forms preserve the base class.",
    ),
    faq(
        "Why can real inputs produce complex output?",
        "A negative real base raised to a noninteger exponent has a principal complex value. The runtime preserves that value instead of returning NaN.",
    ),
    faq(
        "What are the integer exponent rules?",
        "Integer bases require nonnegative integer-valued exponents, preserve the base class, and saturate at the class bounds.",
    ),
    faq(
        "Can scalars and arrays be mixed?",
        "Yes. Scalars expand to the other operand's shape, including supported host-scalar and provider-resident combinations.",
    ),
    faq(
        "What happens if only one input is provider-resident?",
        "Supported scalar forms stay resident. Other forms gather to the host, execute the complete domain rules, and apply any explicit output-prototype request afterward.",
    ),
    faq(
        "Does `power` modify either input?",
        "No. The builtin returns distinct output storage. Fusion may avoid intermediate allocations without changing observable input ownership.",
    ),
    faq(
        "How can I request provider-resident output?",
        "In RunMat mode, pass `'like'` and a provider-resident prototype. The runtime uploads a compatible host result when direct provider execution is unavailable.",
    ),
];
