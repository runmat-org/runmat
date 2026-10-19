use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/control/tf.rs") };

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tf"), slug: Some("tf"),
    summary: "Construct a SISO transfer function from coefficients or a variable shorthand.",
    description: "`tf(num,den)` constructs a scalar-input scalar-output transfer-function object from polynomial coefficients in descending power order. Each input is a numeric scalar, row vector, or column vector; stored coefficient arrays are double-precision row vectors.",
    keywords: &["tf", "transfer function", "control system", "coefficients", "coefficient vector", "polynomial", "feedback", "step response"],
    related: &["ss", "feedback", "step", "roots", "polyval", "damp"],
    sections: &[
        BuiltinDocumentationSection { heading: "Forms and model properties", paragraphs: &[
            "`tf('s')` constructs a continuous-time indeterminate. `tf('z',Ts)` constructs a discrete-time indeterminate with positive Ts or -1 for unspecified sample time. `tf(num,den,Ts)` accepts zero, a positive Ts, or -1; nonzero Ts defaults the variable to z. The `Variable` and `Ts`/`SampleTime` name/value options select the same metadata.",
            "The object stores Numerator, Denominator, Variable, Ts, InputDelay, and OutputDelay. Its class is tf. Coefficients can be real or complex and are kept in highest-power-first order; leading zero coefficients are trimmed. The denominator must contain a nonzero coefficient.",
            "In RunMat mode, p, q, z^-1 and q^-1 variable aliases and `tf('z')` with default Ts = 1 are extensions. Logical and typed-integer coefficients, typed-integer Ts, and explicit gpuArray inputs have independent RunMat-only gates. Integer values must be exactly representable at the binary64 model boundary.",
            "The current constructor builds SISO models, not MIMO coefficient arrays. Separate `tf.*` methods implement scalar SISO arithmetic, including +, -, *, /, left division and integer powers, subject to compatible sample times and unsupported nonzero delays.",
        ] },
        BuiltinDocumentationSection { heading: "Execution and placement", paragraphs: &[
            "Extension admission and option-name validation precede provider access. Supported resident values gather through their owning provider and integer values are checked for exact conversion at the host model boundary. Object construction is a host operation and a fusion boundary, with no GPU transfer-function kernel. The returned value is a new value-semantic object and does not alias input arrays. Gathering can suspend.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "closed-loop", title: "Create a closed-loop model", program: "s = tf('s'); G = 1/(s + 1); T = feedback(2*G, 1); k = dcgain(T); fprintf('%.4f\\n', k);", display_output: Some("0.6667"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(k - 2/3) < 1e-10);" } },
        BuiltinExample { id: "variable", title: "Continuous transfer-function variable", program: "s = tf('s'); G = 2.5/(0.4*s^2 + 1.8*s + 1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(G), 'tf')); assert(max(abs(G.Denominator - [0.4 1.8 1])) < 1e-10);" } },
        BuiltinExample { id: "coefficients", title: "Construct from coefficient vectors", program: "H = tf([1 2], [1 3 2]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(H.Numerator, [1 2])); assert(isequal(H.Denominator, [1 3 2])); assert(H.Ts == 0);" } },
        BuiltinExample { id: "first-order", title: "First-order continuous-time model", program: "H = tf(20, [1 5]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(H), 'tf')); assert(isequal(H.Numerator, 20)); assert(isequal(H.Denominator, [1 5]));" } },
        BuiltinExample { id: "sample-time", title: "Discrete-time model", program: "H = tf(1, [1 -0.5], 0.1);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(H.Ts - 0.1) < 1e-12); assert(strcmp(H.Variable, 'z'));" } },
        BuiltinExample { id: "variable-alias", title: "RunMat variable alias", program: "H = tf([1 0], [1 2 1], 'Variable', 'p');", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(H.Variable, 'p')); assert(isequal(H.Numerator, [1 0]));" } },
        BuiltinExample { id: "integer-extension", title: "Exact integer coefficient boundary", program: "H = tf(uint16([1 2]), int16([1 3 2]));", display_output: None, compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(H.Numerator), 'double')); assert(isequal(H.Numerator, [1 2]));" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does tf evaluate frequency responses?", answer: "No. It constructs a model; frequency-response helpers such as bode, freqresp, and evalfr are separate APIs, not operations performed by this constructor." },
        BuiltinDocumentationFaq { question: "Can I construct models with s = tf('s')?", answer: "Yes. The variable shorthand and integer powers support polynomial model construction." },
        BuiltinDocumentationFaq { question: "Which response and analysis helpers are implemented?", answer: "step, impulse, nyquist, stepinfo, feedback, dcgain, pole, damp, and isstable are separate control builtins." },
        BuiltinDocumentationFaq { question: "Can I pass matrices of coefficients?", answer: "No. Numerator and denominator inputs must be scalars or vectors." },
        BuiltinDocumentationFaq { question: "Can I use integer coefficients?", answer: "RunMat mode accepts all eight integer classes when each coefficient is exactly representable as double. MATLAB compatibility mode restricts coefficients to the documented floating-point surface." },
        BuiltinDocumentationFaq { question: "Can tf build MIMO models?", answer: "Not yet. The current constructor accepts one numerator and one denominator vector." },
        BuiltinDocumentationFaq { question: "What does Ts = -1 mean?", answer: "It marks a discrete-time model with unspecified sample time." },
        BuiltinDocumentationFaq { question: "Can the denominator contain only zeros?", answer: "No. It must include at least one nonzero coefficient." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "ss", target: BuiltinDocumentationLinkTarget::Builtin("ss") },
        BuiltinDocumentationLink { label: "feedback", target: BuiltinDocumentationLinkTarget::Builtin("feedback") },
        BuiltinDocumentationLink { label: "step", target: BuiltinDocumentationLinkTarget::Builtin("step") },
        BuiltinDocumentationLink { label: "roots", target: BuiltinDocumentationLinkTarget::Builtin("roots") },
        BuiltinDocumentationLink { label: "polyval", target: BuiltinDocumentationLinkTarget::Builtin("polyval") },
        BuiltinDocumentationLink { label: "damp", target: BuiltinDocumentationLinkTarget::Builtin("damp") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Constructor forms, coefficient class and shape, compatibility and resident inputs", location: "crates/runmat-runtime/src/builtins/control/tf.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM control workflow", location: "crates/runmat-vm/tests/control.rs" },
    ], notes: &["SISO-only construction and no MIMO coefficient arrays are deliberate partial-coverage limits. Browser, provider and native-artifact tests remain integrated gates."] },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
