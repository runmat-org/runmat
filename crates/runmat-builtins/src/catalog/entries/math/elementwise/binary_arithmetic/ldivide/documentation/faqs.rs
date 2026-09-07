use super::super::super::documentation::faq;
use crate::BuiltinDocumentationFaq;

pub(super) const FAQS: &[BuiltinDocumentationFaq] = &[
    faq(
        "Which input is divided by which?",
        "`ldivide(A, B)` computes `B ./ A`: `A` is the divisor and `B` is the numerator.",
    ),
    faq(
        "Does `ldivide` support implicit expansion?",
        "Yes. Singleton dimensions expand automatically; incompatible non-singleton dimensions produce a size-mismatch error.",
    ),
    faq(
        "What numeric class does `ldivide` return?",
        "Floating-point output is double unless single data participates. Complex inputs produce the corresponding complex floating class. Supported integer forms preserve their integer class.",
    ),
    faq(
        "How does division by zero behave?",
        "Because the call computes `B ./ A`, zero divisors are zero values in `A`. Floating-point results follow IEEE-754 signed-infinity and NaN rules.",
    ),
    faq(
        "Can a provider-resident array be combined with a host scalar?",
        "Yes for supported real scalar forms. The runtime selects scalar division in the correct numerator/denominator direction or uploads an exact scalar in the resident integer class.",
    ),
    faq(
        "What happens when provider execution is unavailable?",
        "RunMat gathers through the owner, evaluates the host contract, and restores explicit device intent. Automatic placement may return a host value when that is the safe result.",
    ),
    faq(
        "How can I request provider-resident output?",
        "In RunMat mode, pass `'like'` and a provider-resident prototype. The result is validated on that provider or the call returns an error.",
    ),
    faq(
        "How are empty arrays handled?",
        "Compatible empty dimensions propagate into the broadcasted output shape, and the result contains no elements.",
    ),
    faq(
        "Are integers and logical values supported?",
        "Logical values enter floating-point division. Matching integer classes, or one integer operand with scalar double, preserve the integer class and use nearest ties-away rounding with saturation.",
    ),
    faq(
        "Can real and complex operands be mixed?",
        "Yes. The result uses the corresponding complex floating class and the same singleton-expansion rules.",
    ),
];
