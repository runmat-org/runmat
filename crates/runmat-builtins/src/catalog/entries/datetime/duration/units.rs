use super::*;
use crate::catalog::inference::{argument_error, finish_fixed};
use runmat_types::{CallInference, CallRequest, ValueKindFact};

const INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor { name: "X", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None, description: "Real numeric/logical array to construct a duration, or duration array to extract fixed-unit counts." }];
const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host duration for numeric/logical input; double counts for duration input.",
}];

macro_rules! descriptor {
    ($id:ident, $label:literal) => {
        const $id: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: &INPUT,
                outputs: &OUTPUT,
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: errors::UNIT_ERRORS,
        };
    };
}
descriptor!(DAYS, "out = days(X)");
descriptor!(HOURS, "out = hours(X)");
descriptor!(MINUTES, "out = minutes(X)");
descriptor!(SECONDS, "out = seconds(X)");
descriptor!(MILLISECONDS, "out = milliseconds(X)");
descriptor!(YEARS, "out = years(X)");

const INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability { name: "X", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Integer input is converted at the binary64 fixed-day-count construction boundary; duration input extracts double counts." }];
const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor { form: "durationUnit(integer_X)", inputs: &INTEGER_INPUT, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving, notes: "All eight integer classes are accepted; finite input that overflows the host day count is rejected. The returned duration keeps array shape." }];
const UNIT_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [constructor::DURATION_GPU_INPUT_EXTENSION];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Duration unit implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/duration/model.rs") };
const TESTS: &[BuiltinEvidenceReference] = &[BuiltinEvidenceReference {
    kind: BuiltinEvidenceKind::UnitTest,
    label: "Duration units and conversion",
    location: "crates/runmat-runtime/src/builtins/duration/tests.rs",
}];

macro_rules! unit_documentation {
    ($id:ident, $name:literal, $unit:literal, $program:literal, $assertions:literal, $url:literal) => {
        const $id: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog, title: Some($name), slug: Some($name),
            summary: concat!("Create fixed durations from ", $unit, " or extract ", $unit, " from a duration."),
            description: concat!("`", $name, "(X)` creates a host duration from real numeric or logical X. For duration input it returns double ", $unit, " counts. Array shape is retained in either direction."),
            keywords: &[$name, "duration", "fixed elapsed time", "unit conversion"], related: &["duration", "datetime"],
            sections: &[
                BuiltinDocumentationSection { heading: "Values and conversion", paragraphs: &["Numeric classes enter binary64 day-count storage at construction. NaN and positive or negative infinity remain nonfinite durations; finite overflow is rejected. Logical values represent zero and one. Extracting a duration returns double without changing the duration.", "A year here is a fixed 365.2425-day duration, not a calendar year. Use calendarDuration and calyears for calendar-relative years."] },
                BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The result is host-side and forms a fusion boundary; no unit-specific GPU kernel runs. Resident numeric input is a RunMat-only extension, admitted before owner-resolved download. The same host path is available in native and WebAssembly runtimes."] },
            ],
            examples: &[BuiltinExample { id: "roundtrip", title: concat!("Construct and extract ", $unit), program: $program, display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: $assertions } }],
            example_exemption: None, faqs: &[BuiltinDocumentationFaq { question: "Is the result stored on the GPU?", answer: "No. Duration objects and extracted counts are host-side. RunMat mode can gather resident numeric input through its owning provider." }],
            links: &[BuiltinDocumentationLink { label: "MATLAB unit reference", target: BuiltinDocumentationLinkTarget::External($url) }, BuiltinDocumentationLink { label: "duration", target: BuiltinDocumentationLinkTarget::Builtin("duration") }, SOURCE],
            media: &[], evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: TESTS, notes: &["Tall and distributed input variants remain outside this host duration implementation."] },
            introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
        };
    }
}

unit_documentation!(
    DAYS_DOC,
    "days",
    "days",
    "d = days([1 2]); n = days(d);",
    "assert(isduration(d)); assert(isequal(n,[1 2]));",
    "https://www.mathworks.com/help/matlab/ref/duration.days.html"
);
unit_documentation!(
    HOURS_DOC,
    "hours",
    "hours",
    "d = hours([1 2]); n = hours(d);",
    "assert(isduration(d)); assert(isequal(n,[1 2]));",
    "https://www.mathworks.com/help/matlab/ref/duration.hours.html"
);
unit_documentation!(
    MINUTES_DOC,
    "minutes",
    "minutes",
    "d = minutes([1 2]); n = minutes(d);",
    "assert(isduration(d)); assert(isequal(n,[1 2]));",
    "https://www.mathworks.com/help/matlab/ref/duration.minutes.html"
);
unit_documentation!(
    SECONDS_DOC,
    "seconds",
    "seconds",
    "d = seconds([1 2]); n = seconds(d);",
    "assert(isduration(d)); assert(isequal(n,[1 2]));",
    "https://www.mathworks.com/help/matlab/ref/duration.seconds.html"
);
unit_documentation!(
    MILLISECONDS_DOC,
    "milliseconds",
    "milliseconds",
    "d = milliseconds([1 2]); n = milliseconds(d);",
    "assert(isduration(d)); assert(all(abs(n-[1 2]) < 1e-12));",
    "https://www.mathworks.com/help/matlab/ref/duration.milliseconds.html"
);
unit_documentation!(
    YEARS_DOC,
    "years",
    "years",
    "d = years([1 2]); n = years(d);",
    "assert(isduration(d)); assert(all(abs(n-[1 2]) < 1e-12));",
    "https://www.mathworks.com/help/matlab/ref/duration.years.html"
);

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-DURATION-UNIT-ARITY",
            format!("{} expects one input", entry.identity.name),
            0,
        ));
    }
    let output = match request.arguments.first() {
        Some(input) if facts::duration_class(input) == Some(true) => {
            facts::double(input.shape.clone())
        }
        Some(input)
            if matches!(
                input.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) && facts::real_numeric_or_logical(input) =>
        {
            facts::duration(input.shape.clone())
        }
        Some(input) => {
            if facts::duration_class(input) == Some(false) && !facts::real_numeric_or_logical(input)
            {
                diagnostics.push(argument_error(
                    "RM-DURATION-UNIT-INPUT",
                    "expected a real numeric, logical, or duration value",
                    0,
                ));
            }
            facts::unknown_host(input.shape.clone())
        }
        None => facts::unknown_host(runmat_types::ShapeFact::Unknown),
    };
    finish_fixed(entry, request, output, diagnostics)
}

macro_rules! unit_entry {
    ($id:ident, $name:literal, $desc:ident, $doc:ident) => {
        pub const $id: BuiltinCatalogEntry = entry(
            $name,
            &$desc,
            $doc,
            infer,
            true,
            &UNIT_EXTENSIONS,
            &INTEGER_CAPABILITIES,
        );
    };
}
unit_entry!(DAYS_CATALOG_ENTRY, "days", DAYS, DAYS_DOC);
unit_entry!(HOURS_CATALOG_ENTRY, "hours", HOURS, HOURS_DOC);
unit_entry!(MINUTES_CATALOG_ENTRY, "minutes", MINUTES, MINUTES_DOC);
unit_entry!(SECONDS_CATALOG_ENTRY, "seconds", SECONDS, SECONDS_DOC);
unit_entry!(
    MILLISECONDS_CATALOG_ENTRY,
    "milliseconds",
    MILLISECONDS,
    MILLISECONDS_DOC
);
unit_entry!(YEARS_CATALOG_ENTRY, "years", YEARS, YEARS_DOC);
