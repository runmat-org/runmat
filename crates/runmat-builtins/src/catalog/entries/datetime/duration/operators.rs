use super::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use runmat_types::{CallInference, CallRequest, ShapeFact};

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left duration operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Duration or numeric fixed-day count; plus also accepts datetime.",
    },
];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical comparison, duration arithmetic, or datetime after addition.",
}];
pub const DURATION_BINARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "out = duration.op(lhs, rhs)",
        inputs: &INPUTS,
        outputs: &OUTPUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::MethodOnly,
    errors: errors::OPERATOR_ERRORS,
};
const EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [constructor::DURATION_GPU_INPUT_EXTENSION];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Duration operators", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/duration/builtins.rs") };
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Duration comparison and arithmetic",
    location: "crates/runmat-runtime/src/builtins/duration/tests.rs",
}];

macro_rules! operator_doc {
    ($id:ident, $name:literal, $summary:literal, $program:literal, $oracle:literal, $url:literal) => {
        const $id: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog, title: Some($name), slug: Some($name), summary: $summary,
            description: concat!("Duration method `", $name, "` applies elementwise to fixed elapsed-time values. A real numeric operand represents a count of 24-hour days. Scalars expand over arrays; compatible dimensions broadcast. The result stays on the host."),
            keywords: &["duration", $name, "elapsed time", "operator"], related: &["duration", "days", "datetime"],
            sections: &[BuiltinDocumentationSection { heading: "Array and runtime behavior", paragraphs: &["Duration storage is binary64 day counts. Comparison uses exact stored values, including ordinary NaN comparison rules; no tolerance is applied. Arithmetic retains duration for duration/numeric operands, while duration plus datetime returns datetime.", "Resident consumed numeric payloads are admitted in RunMat mode before owner-resolved download. Strict compatibility rejects that extension before provider access. These methods do not launch GPU kernels or perform workspace, file, or network effects."] }],
            examples: &[BuiltinExample { id: "operator", title: $summary, program: $program, display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: $oracle } }],
            example_exemption: None, faqs: &[], links: &[BuiltinDocumentationLink { label: "MATLAB operator reference", target: BuiltinDocumentationLinkTarget::External($url) }, BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Builtin("duration") }, SOURCE],
            media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Text operands and some MATLAB-supported distributed/tall forms remain unsupported."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
        };
    };
}
operator_doc!(
    EQ_DOC,
    "duration.eq",
    "Compare elapsed times for equality.",
    "a = seconds(2); b = seconds(2); tf = a == b;",
    "assert(islogical(tf)); assert(tf);",
    "https://www.mathworks.com/help/matlab/ref/duration.eq.html"
);
operator_doc!(
    NE_DOC,
    "duration.ne",
    "Compare elapsed times for inequality.",
    "a = seconds(2); b = seconds(3); tf = a ~= b;",
    "assert(islogical(tf)); assert(tf);",
    "https://www.mathworks.com/help/matlab/ref/duration.ne.html"
);
operator_doc!(
    LT_DOC,
    "duration.lt",
    "Compare elapsed times with less-than.",
    "a = seconds(2); b = seconds(3); tf = a < b;",
    "assert(islogical(tf)); assert(tf);",
    "https://www.mathworks.com/help/matlab/ref/duration.lt.html"
);
operator_doc!(
    LE_DOC,
    "duration.le",
    "Compare elapsed times with less-than-or-equal.",
    "a = seconds(2); b = seconds(2); tf = a <= b;",
    "assert(islogical(tf)); assert(tf);",
    "https://www.mathworks.com/help/matlab/ref/duration.le.html"
);
operator_doc!(
    GT_DOC,
    "duration.gt",
    "Compare elapsed times with greater-than.",
    "a = seconds(3); b = seconds(2); tf = a > b;",
    "assert(islogical(tf)); assert(tf);",
    "https://www.mathworks.com/help/matlab/ref/duration.gt.html"
);
operator_doc!(
    GE_DOC,
    "duration.ge",
    "Compare elapsed times with greater-than-or-equal.",
    "a = seconds(2); b = seconds(2); tf = a >= b;",
    "assert(islogical(tf)); assert(tf);",
    "https://www.mathworks.com/help/matlab/ref/duration.ge.html"
);
operator_doc!(
    PLUS_DOC,
    "duration.plus",
    "Add fixed elapsed times or shift a datetime.",
    "a = hours(2); b = a + hours(3);",
    "assert(isduration(b)); assert(hours(b)==5);",
    "https://www.mathworks.com/help/matlab/ref/duration.plus.html"
);
operator_doc!(
    MINUS_DOC,
    "duration.minus",
    "Subtract fixed elapsed times.",
    "a = hours(5); b = a - hours(2);",
    "assert(isduration(b)); assert(hours(b)==3);",
    "https://www.mathworks.com/help/matlab/ref/duration.minus.html"
);

fn output_shape(
    request: &CallRequest,
    diagnostics: &mut Vec<runmat_types::InferenceDiagnostic>,
) -> ShapeFact {
    match request.arguments.as_slice() {
        [lhs, rhs] => match runmat_types::broadcast_shape(&lhs.shape, &rhs.shape) {
            Ok(shape) => shape,
            Err(error) => {
                diagnostics.push(error);
                ShapeFact::Unknown
            }
        },
        _ => ShapeFact::Unknown,
    }
}
fn infer_comparison(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-DURATION-OP-ARITY",
            "duration comparison expects two operands",
            0,
        ));
    }
    if request.arguments.len() == 2
        && request
            .arguments
            .iter()
            .all(|value| facts::duration_class(value) == Some(false))
    {
        diagnostics.push(argument_error(
            "RM-DURATION-OP-RECEIVER",
            "at least one comparison operand must be a duration",
            0,
        ));
    }
    for (index, value) in request.arguments.iter().enumerate() {
        if facts::duration_class(value) == Some(false) && !facts::real_numeric_or_logical(value) {
            diagnostics.push(argument_error(
                "RM-DURATION-OP-INPUT",
                "expected duration or real numeric day count",
                index,
            ));
        }
    }
    let shape = output_shape(request, &mut diagnostics);
    finish_fixed(entry, request, facts::logical(shape), diagnostics)
}
fn infer_plus(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_arithmetic(request, entry, BinaryArithmeticInferenceRule::Add)
}
fn infer_minus(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_arithmetic(request, entry, BinaryArithmeticInferenceRule::Subtract)
}
fn infer_arithmetic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    operation: BinaryArithmeticInferenceRule,
) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-DURATION-OP-ARITY",
            "duration arithmetic expects two operands",
            0,
        ));
    }
    if let Some(lhs) = request.arguments.first() {
        if facts::duration_class(lhs) == Some(false) {
            diagnostics.push(argument_error(
                "RM-DURATION-OP-LEFT",
                "expected duration receiver",
                0,
            ));
        }
    }
    let shape = output_shape(request, &mut diagnostics);
    let output = if request.arguments.len() == 2 {
        let rhs = &request.arguments[1];
        match &rhs.kind {
            runmat_types::ValueKindFact::Object(object)
                if operation == BinaryArithmeticInferenceRule::Add
                    && object.runtime_class.as_ref()
                        == Some(&runmat_types::standard::DATETIME.owned()) =>
            {
                facts::datetime(shape)
            }
            runmat_types::ValueKindFact::Unknown => facts::unknown_host(shape),
            runmat_types::ValueKindFact::Object(object)
                if object.runtime_class.is_none() && object.class.is_none() =>
            {
                facts::unknown_host(shape)
            }
            runmat_types::ValueKindFact::Object(_) if facts::duration_class(rhs) == Some(false) => {
                diagnostics.push(argument_error(
                    "RM-DURATION-OP-RIGHT",
                    "expected duration, real numeric day count, or datetime for plus",
                    1,
                ));
                facts::unknown_host(shape)
            }
            _ if facts::duration_class(rhs) == Some(false)
                && !facts::real_numeric_or_logical(rhs) =>
            {
                diagnostics.push(argument_error(
                    "RM-DURATION-OP-RIGHT",
                    "expected duration or real numeric day count",
                    1,
                ));
                facts::unknown_host(shape)
            }
            _ => facts::duration(shape),
        }
    } else {
        facts::unknown_host(shape)
    };
    finish_fixed(entry, request, output, diagnostics)
}
macro_rules! cmp_entry {
    ($id:ident, $name:literal, $doc:ident) => {
        pub const $id: BuiltinCatalogEntry = entry(
            $name,
            &DURATION_BINARY_DESCRIPTOR,
            $doc,
            infer_comparison,
            true,
            &EXTENSIONS,
            &[],
        );
    };
}
cmp_entry!(DURATION_EQ_CATALOG_ENTRY, "duration.eq", EQ_DOC);
cmp_entry!(DURATION_NE_CATALOG_ENTRY, "duration.ne", NE_DOC);
cmp_entry!(DURATION_LT_CATALOG_ENTRY, "duration.lt", LT_DOC);
cmp_entry!(DURATION_LE_CATALOG_ENTRY, "duration.le", LE_DOC);
cmp_entry!(DURATION_GT_CATALOG_ENTRY, "duration.gt", GT_DOC);
cmp_entry!(DURATION_GE_CATALOG_ENTRY, "duration.ge", GE_DOC);
pub const DURATION_PLUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "duration.plus",
    &DURATION_BINARY_DESCRIPTOR,
    PLUS_DOC,
    infer_plus,
    true,
    &EXTENSIONS,
    &[],
);
pub const DURATION_MINUS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "duration.minus",
    &DURATION_BINARY_DESCRIPTOR,
    MINUS_DOC,
    infer_minus,
    true,
    &EXTENSIONS,
    &[],
);
