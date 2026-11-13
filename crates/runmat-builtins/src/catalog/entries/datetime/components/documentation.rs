use crate::*;

mod year;
mod month;
mod day;
mod hour;
mod minute;
pub(super) use day::DOCUMENTATION as DAY;
pub(super) use hour::DOCUMENTATION as HOUR;
pub(super) use minute::DOCUMENTATION as MINUTE;
pub(super) use month::DOCUMENTATION as MONTH;
pub(super) use year::DOCUMENTATION as YEAR;

macro_rules! documentation {
    ($constant:ident, $authority:ident, $name:literal, $summary:literal, $behavior:literal, $program:literal, $assertions:literal) => {
        pub(super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::$authority,
            title: Some($name),
            slug: Some($name),
            summary: $summary,
            description: $behavior,
            keywords: &[$name, "datetime", "calendar"],
            related: &["datetime", "datenum", "datevec"],
            sections: &[BuiltinDocumentationSection {
                heading: "Clock and compatibility",
                paragraphs: &[
                    "Current and relative date text uses the invocation-owned calendar service. Hosted execution uses the admitted local platform calendar; deterministic deployment uses its explicit calendar policy and fails closed when unavailable.",
                    "Component numeric results are host doubles. Month and day name modes on datetime values return cell arrays of character vectors. Legacy serial/date-text forms retain their documented input-format and extension gates; this is a partial MATLAB surface.",
                ],
            }],
            examples: &[BuiltinExample {
                id: concat!($name, "-calendar"),
                title: $summary,
                program: $program,
                display_output: None,
                compatibility: BuiltinExampleCompatibility::Matlab,
                harness: BuiltinExampleHarness::Native,
                fixture: BuiltinExampleFixture::None,
                requirements: BuiltinExampleRequirements::NONE,
                verification: BuiltinExampleVerification::Assertions { source: $assertions },
            }],
            example_exemption: None,
            faqs: &[],
            links: &[BuiltinDocumentationLink {
                label: "datetime",
                target: BuiltinDocumentationLinkTarget::Builtin("datetime"),
            }],
            media: &[],
            evidence: BuiltinDocumentationEvidence {
                implementation: &[BuiltinDocumentationLink {
                    label: "Calendar component and current-time implementation",
                    target: BuiltinDocumentationLinkTarget::Source(
                        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/datetime/component_builtins.rs",
                    ),
                }],
                verification: &[BuiltinEvidenceReference {
                    kind: BuiltinEvidenceKind::UnitTest,
                    label: "Canonical calendar facts and Clock contracts",
                    location: "crates/runmat-builtins/src/catalog/entries/datetime/components/tests.rs",
                }],
                notes: &[],
            },
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Partial),
        };
    };
}

documentation!(NOW, Catalog, "now", "Read the current local serial date number.",
    "now() takes no inputs and returns a scalar double serial date number, including the local time of day. It reads the admitted calendar service once; it does not create an independent clock.",
    "n = now();",
    "assert(isscalar(n) && isfinite(n));");
documentation!(TODAY, Catalog, "today", "Read today's local date as a serial number or datetime.",
    "today() and today('datenum') return a scalar double serial date at local midnight. today('datetime') returns that local date as a scalar datetime using the default date format. The optional outputType accepts scalar string or character text. A successful call reads the admitted calendar service once after validating the option.",
    "n = today();\nd = today('datenum');\nt = today('datetime');",
    "assert(isscalar(n) && isfinite(n) && n == floor(n));\nassert(isscalar(d) && isfinite(d) && d == floor(d));\nassert(isdatetime(t));\nassert(hour(t) == 0 && minute(t) == 0);");
documentation!(CLOCK, Catalog, "clock", "Read the current local date vector.",
    "clock() takes no inputs and returns a 1-by-6 host double vector [year month day hour minute second]. Fractional seconds are retained. The owned calendar service controls current time and local conversion.",
    "c = clock();",
    "assert(isequal(size(c), [1 6]));\nassert(c(2) >= 1 && c(2) <= 12);");
