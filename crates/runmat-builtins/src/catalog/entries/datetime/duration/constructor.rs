use super::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use runmat_types::{
    CallInference, CallRequest, DimensionFact, NumericDomain, ShapeFact, ValueKindFact,
};

pub const DURATION_SHORT_COMPONENT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "duration-short-component-form",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "One- and two-component numeric duration construction is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:DurationShortComponentFormExtension"),
    };
pub const DURATION_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "duration-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "Resident numeric input to duration construction, unit conversion, and duration methods is gathered only in RunMat mode",
    error_identifier: Some("RunMat:compatibility:DurationGpuInputExtension"),
};
pub const DURATION_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DURATION_SHORT_COMPONENT_EXTENSION,
    DURATION_GPU_INPUT_EXTENSION,
];

const OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host duration array.",
}];
const X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-by-3 matrix of hour, minute, and second components.",
}];
const fn component(name: &'static str, description: &'static str) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
const H: [BuiltinParamDescriptor; 1] = [component(
    "hours",
    "Hours (RunMat-only one-component form).",
)];
const HM: [BuiltinParamDescriptor; 2] = [
    component("hours", "Hours."),
    component("minutes", "Minutes."),
];
const HMS: [BuiltinParamDescriptor; 3] = [
    component("hours", "Hours."),
    component("minutes", "Minutes."),
    component("seconds", "Seconds."),
];
const HMSM: [BuiltinParamDescriptor; 4] = [
    component("hours", "Hours."),
    component("minutes", "Minutes."),
    component("seconds", "Seconds."),
    component("milliseconds", "Milliseconds."),
];
const VARARGS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "args",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Numeric components followed by an optional Format name-value pair.",
}];
pub const DURATION_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "t = duration(X)",
            inputs: &X,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = duration(hours)",
            inputs: &H,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = duration(hours, minutes)",
            inputs: &HM,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = duration(hours, minutes, seconds)",
            inputs: &HMS,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = duration(hours, minutes, seconds, milliseconds)",
            inputs: &HMSM,
            outputs: &OUT,
        },
        BuiltinSignatureDescriptor {
            label: "t = duration(___, \"Format\", format)",
            inputs: &VARARGS,
            outputs: &OUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: errors::CONSTRUCTOR_ERRORS,
};

const INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "components or X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight fixed-width classes enter the documented floating day-count construction boundary.",
}];
pub const DURATION_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "duration(integer_H,integer_MI,integer_S,integer_MS?)", inputs: &INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Scalar components expand; matching non-scalar shapes are required. Output is a host duration object backed by binary64 day counts." },
    BuiltinIntegerCapabilityDescriptor { form: "duration(integer_X)", inputs: &INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Each row of an N-by-3 matrix yields one duration in an N-by-1 host array." },
];

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Duration implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/duration/mod.rs") };
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Duration construction and methods",
    location: "crates/runmat-runtime/src/builtins/duration/tests.rs",
}];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("duration"), slug: Some("duration"),
    summary: "Create fixed elapsed-time arrays from real numeric or logical components or a three-column matrix.",
    description: "`duration(H,MI,S)` and `duration(H,MI,S,MS)` construct host duration arrays; `duration(X)` interprets each row of an N-by-3 real numeric or logical matrix as hours, minutes, and seconds. Logical values count as zero or one. The elapsed value is stored as binary64 days. The Format property changes rendering, not arithmetic or comparisons.",
    keywords: &["duration", "elapsed time", "time span", "Format", "MATLAB compatibility"],
    related: &["datetime", "calendarDuration", "days", "hours", "string", "char", "disp"],
    sections: &[
        BuiltinDocumentationSection { heading: "Numeric construction", paragraphs: &[
            "All eight fixed-width integer classes and logical values are accepted in component forms, with conversion at the binary64 day-count boundary. Scalar components expand to the shape of non-scalars; non-scalars must have identical shape. Empty and N-D shapes are retained. NaN and positive/negative infinity components remain nonfinite elapsed values and render as NaN, Inf, and -Inf.",
            "An N-by-3 matrix makes an N-by-1 array. The one- and two-component conveniences are RunMat-only forms. Text parsing forms are not implemented. Supported display formats are hh:mm:ss, hh:mm, mm:ss, s, and ss.",
        ]},
        BuiltinDocumentationSection { heading: "Methods and placement", paragraphs: &[
            "The host object stores a day-count tensor and Format property. Parentheses indexing preserves the duration class; Format can be assigned without changing elapsed values. Comparisons and plus/minus are elementwise; a numeric operand denotes fixed 24-hour days. Duration plus datetime shifts the datetime. Calendar-relative months belong to calendarDuration, not duration.",
            "Resident numeric input is a RunMat-only extension. It is checked before provider access, gathered through its owner, and produces a host duration. There is no duration GPU kernel or fusion; native and WebAssembly runtimes use the same host semantics.",
        ]},
    ],
    examples: &[
        BuiltinExample { id: "components", title: "Construct a scalar", program: "t = duration(1,30,45); h = hours(t); fprintf('t =\\n'); disp(t);", display_output: Some("t =\n  01:30:45"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isduration(t)); assert(abs(h - (1+30/60+45/3600)) < 1e-12);" } },
        BuiltinExample { id: "broadcast", title: "Expand a scalar component", program: "t = duration([1 2],15,0); h = hours(t); fprintf('t =\\n'); disp(t);", display_output: Some("t =\n01:15:00  02:15:00"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(t),[1 2])); assert(all(abs(h - [1.25 2.25]) < 1e-12));" } },
        BuiltinExample { id: "format", title: "Change rendering only", program: "t = duration(1,5,0); t.Format = 'hh:mm'; h = hours(t); disp(t)", display_output: Some("01:05"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(h - (1+5/60)) < 1e-12); assert(strcmp(t.Format,'hh:mm'));" } },
        BuiltinExample { id: "datetime", title: "Shift a datetime", program: "t0 = datetime(2024,4,9,8,0,0); t1 = t0 + duration(1,30,0); fprintf('t1 =\\n'); disp(t1);", display_output: Some("t1 =\n  09-Apr-2024 09:30:00"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(hour(t1)==9); assert(minute(t1)==30);" } },
        BuiltinExample { id: "units", title: "Convert fixed units", program: "d = days(1); h = hours(d); fprintf('h = %g\\n', h);", display_output: Some("h = 24"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isduration(d)); assert(h==24);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What does Format change?", answer: "Only rendering. Stored elapsed time and comparisons are unchanged." },
        BuiltinDocumentationFaq { question: "Can I index duration arrays?", answer: "Yes. Parentheses indexing preserves the duration class; Format is a separate property." },
        BuiltinDocumentationFaq { question: "Does duration run on a GPU?", answer: "Duration objects are host-side. In RunMat mode resident numeric input may be gathered through its owner after compatibility admission." },
        BuiltinDocumentationFaq { question: "Can I add duration to datetime?", answer: "Yes. The result is a datetime shifted by the fixed elapsed time." },
    ],
    links: &[BuiltinDocumentationLink { label: "MATLAB duration reference", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/duration.html") }, BuiltinDocumentationLink { label: "datetime", target: BuiltinDocumentationLinkTarget::Builtin("datetime") }, BuiltinDocumentationLink { label: "calendarDuration", target: BuiltinDocumentationLinkTarget::Builtin("calendarDuration") }, BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") }, BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") }, BuiltinDocumentationLink { label: "disp", target: BuiltinDocumentationLinkTarget::Builtin("disp") }, SOURCE],
    media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Text construction and some documented MATLAB formats and distributed/tall forms remain unsupported."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut positional_end = request.arguments.len();
    let mut unresolved_options = false;
    while positional_end >= 3 {
        let option_index = positional_end - 2;
        let Some(option) = request.literals.literal_string_at(option_index) else {
            if matches!(
                request.arguments[option_index].kind,
                ValueKindFact::String | ValueKindFact::Character
            ) {
                unresolved_options = true;
                positional_end -= 2;
            }
            break;
        };
        if option != "format" {
            diagnostics.push(argument_error(
                "RM-DURATION-OPTION",
                format!("unsupported duration option '{option}'"),
                option_index,
            ));
            break;
        }
        if !matches!(
            request.arguments[positional_end - 1].kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-DURATION-FORMAT",
                "Format requires a text scalar",
                positional_end - 1,
            ));
        }
        positional_end -= 2;
    }
    let count = positional_end;
    if !(1..=4).contains(&count) && !unresolved_options {
        diagnostics.push(argument_error(
            "RM-DURATION-ARITY",
            "duration expects a three-column matrix or numeric components",
            0,
        ));
    }
    let mut shape = ShapeFact::Scalar;
    if count == 1 {
        let input = &request.arguments[0];
        shape = match input.shape.known_dims() {
            Some(dims) if dims.len() == 2 && dims[1] == Some(3) => ShapeFact::Shaped {
                dims: vec![
                    dims[0].map_or(DimensionFact::Unknown, DimensionFact::Known),
                    DimensionFact::Known(1),
                ],
            },
            _ => input.shape.clone(),
        };
    } else {
        let mut first_non_scalar: Option<&ShapeFact> = None;
        for (index, input) in request.arguments[..count].iter().enumerate() {
            if !matches!(
                input.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(runmat_types::NumericFact {
                        domain: NumericDomain::Real,
                        ..
                    })
            ) {
                diagnostics.push(argument_error(
                    "RM-DURATION-INPUT",
                    "expected real numeric or logical component",
                    index,
                ));
            }
            if input.shape.element_count() != Some(1) {
                if let Some(first) = first_non_scalar {
                    let known_shape = |shape: &ShapeFact| {
                        shape
                            .known_dims()
                            .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
                    };
                    if let (Some(left), Some(right)) =
                        (known_shape(first), known_shape(&input.shape))
                    {
                        if left != right {
                            diagnostics.push(argument_error(
                                "RM-DURATION-COMPONENT-SHAPE",
                                "non-scalar duration components must have matching sizes",
                                index,
                            ));
                        }
                    }
                    if shape != input.shape {
                        shape = ShapeFact::Unknown;
                    }
                } else {
                    first_non_scalar = Some(&input.shape);
                    shape = input.shape.clone();
                }
            }
        }
    }
    if count == 1
        && !matches!(
            request.arguments[0].kind,
            ValueKindFact::Unknown
                | ValueKindFact::Logical
                | ValueKindFact::Numeric(runmat_types::NumericFact {
                    domain: NumericDomain::Real,
                    ..
                })
        )
    {
        diagnostics.push(argument_error(
            "RM-DURATION-INPUT",
            "expected a real numeric or logical three-column matrix",
            0,
        ));
    }
    finish_fixed(entry, request, facts::duration(shape), diagnostics)
}

pub const DURATION_CATALOG_ENTRY: BuiltinCatalogEntry = entry(
    "duration",
    &DURATION_DESCRIPTOR,
    DOCUMENTATION,
    infer,
    true,
    &DURATION_EXTENSIONS,
    &DURATION_INTEGER_CAPABILITIES,
);
