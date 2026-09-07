use super::super::super::documentation::faq;
use crate::BuiltinDocumentationFaq;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    faq(
        "Does `rdivide` support implicit expansion?",
        "Yes. Singleton dimensions expand automatically. Incompatible non-singleton dimensions produce a size-mismatch error.",
    ),
    faq(
        "What numeric class does `rdivide` return?",
        "Floating-point output is double unless single data participates. Complex inputs produce the corresponding complex floating class. Supported integer forms preserve their integer class; logical and character inputs enter floating-point arithmetic.",
    ),
    faq(
        "How does division by zero behave?",
        "Floating-point division follows IEEE-754: nonzero finite values divided by zero produce signed infinity, and zero divided by zero produces NaN. Complex division uses the complex quotient.",
    ),
    faq(
        "Can a provider-resident array be divided by a host scalar?",
        "Yes for supported real scalar forms. RunMat uses the provider's scalar operation or an exact typed scalar upload when the contract permits it.",
    ),
    faq(
        "What happens when provider execution is unavailable?",
        "RunMat gathers through the owning provider and evaluates the host contract. Explicit device intent must be restored; automatically placed values may remain on the host.",
    ),
    faq(
        "How can I request provider-resident output?",
        "In RunMat mode, pass `'like'` and a provider-resident prototype. The runtime either returns a validated result on that provider or reports that the request cannot be preserved.",
    ),
    faq(
        "How are empty arrays handled?",
        "Compatible empty dimensions propagate into the broadcasted output shape, and the result contains no elements.",
    ),
    faq(
        "Are fixed-width integers supported?",
        "Yes. Matching integer classes, or one integer operand with scalar double, preserve the integer class and use nearest ties-away rounding with saturation.",
    ),
    faq(
        "Can real and complex operands be mixed?",
        "Yes. The result is complex, with the same singleton-expansion rules as real division.",
    ),
    faq(
        "Are string arrays accepted?",
        "No. String arrays are not numeric operands for `rdivide` and produce an input error.",
    ),
];
