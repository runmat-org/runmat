use super::super::moving_window_contract::{
    MOVING_WINDOW_CLASSES_TEXT, MOVING_WINDOW_CONTROLS_TEXT, MOVING_WINDOW_ERRORS_TEXT,
    MOVING_WINDOW_PERFORMANCE_TEXT,
};
use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("movmean"),
    slug: Some("movmean"),
    summary: "Compute moving averages in MATLAB and RunMat.",
    description: "`movmean(A,k)` computes means over a sliding window of length `k` along the first non-singleton dimension.",
    keywords: &["movmean", "moving", "window", "mean", "omitnan"],
    related: &["mean", "movsum"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &["`k` may be a positive scalar window length or a two-element `[kb kf]` backward/forward window.", "Pass an explicit `dim` to move along a chosen dimension.", "`'includenan'` is the default; `'omitnan'` and `'omitmissing'` skip NaN values inside each window.", "`'Endpoints'` supports `'shrink'` (default), `'discard'`, `'fill'`, and numeric scalar fill values.", "`'SamplePoints', sp` uses a strictly increasing numeric point vector for irregular moving windows. With sample points, RunMat currently supports shrink endpoints.", "This implementation covers numeric, logical, complex, and gpuArray inputs; table/timetable `DataVariables` forms are tracked separately."],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[concat!("Real dense gpuArray inputs use provider-resident count-window reductions for scalar or [kb kf] windows, explicit dimensions, omit/include NaN modes, shrink/discard/fill endpoints, and sample/population normalization for ", "movstd/movvar. SamplePoints and unsupported provider backends fall back to gather-once host evaluation and re-upload when an acceleration provider is active; complex GPU moving-window support remains tracked outside this ", "resident real-dense fast path. These host fallback statements apply only to admitted Unbounded host evaluation and compatibility-enabled forms. Complex GPU input is outside the implemented resident path; the statements ", "do not certify a complex-GPU fallback. Provider failures after an attempt do not select fallback. See the precision, compatibility, and finite-allocation boundaries below.",)],
        },
        BuiltinDocumentationSection {
            heading: "Accelerator scope",
            paragraphs: &["GPU acceleration is optional: real dense count-window kernels reduce overlapping windows rather than applying an elementwise operation. Scalar and two-element window controls, explicit dimensions and endpoint geometry follow the documented MATLAB-style forms; this is not a general binary-array broadcasting guarantee.", "f32: native real single precision is supported by WGPU moving-window kernels; a provider must return physical single output for single input.", "f64: native real double precision is supported by WGPU moving-window kernels and the current SimpleProvider reference hook."],
        },
        BuiltinDocumentationSection {
            heading: "Classes and storage",
            paragraphs: &[MOVING_WINDOW_CLASSES_TEXT],
        },
        BuiltinDocumentationSection {
            heading: "Compatibility and errors",
            paragraphs: &[concat!(
                "Compatibility is limited to the documented count/span windows and controls, not complete MATLAB parity. Table/timetable DataVariables forms are not implemented. GPU SamplePoints is a compatibility-gated gather/re-upload ",
                "extension where host fallback is admitted. movmedian GPU count windows larger than 31 require its existing large-window compatibility extension; the WGPU native rank-selection implementation supports at most 256 ",
                "elements. Explicit GPU data and controls are never silently gathered: unsupported explicit forms require the caller to gather first, and enabling a language compatibility extension does not grant placement authority. ",
                "Automatic-provenance provider refusals may select the existing host fallback only when the canonical typed unsupported/no-attempt predicate allows it. Execution, cancellation, stale-authority, malformed-result, and ",
                "restore failures are terminal; original typed causes are retained and no host replay is permitted after a provider attempt.",
            )],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &[concat!("Host evaluation, SamplePoints scratch, and gather/re-upload fallback currently require Unbounded host allocation. Under a Managed finite policy the shared moving-window admission fence refuses before unowned host ", "parsing, cloning, gathering, or numerical scratch. A supported resident real provider request is not refused merely because host fallback is unavailable: rooted control preparation and its exact borrowed request-shape ", "workspace use existing canonical claims, checked before and after suspension and before publication. Provider execution still depends on the provider's own allocation/admission contract; the SimpleProvider hook has ", "ordinary private host scratch. Guarded finite host refusal and claimed request metadata are not completion of finite moving-kernel workspace ownership (L25).",)],
        },
        BuiltinDocumentationSection {
            heading: "Performance",
            paragraphs: &[MOVING_WINDOW_PERFORMANCE_TEXT],
        },
        BuiltinDocumentationSection {
            heading: "Complete control grammar",
            paragraphs: &[MOVING_WINDOW_CONTROLS_TEXT],
        },
        BuiltinDocumentationSection {
            heading: "Error and authority boundaries",
            paragraphs: &[MOVING_WINDOW_ERRORS_TEXT],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Moving mean across columns",
            program: "A = [1 3 5; 2 4 6];\nM = movmean(A,3,2)",
            display_output: Some("M = [2 3 4; 3 4 5]"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(M, [2 3 4; 3 4 5]));",
            },
        },
        BuiltinExample {
            id: "native-complex-single",
            title: "Preserve complex single moving means",
            program: "A = complex(single([1 3 5]), single([1 1 1]));\nM = movmean(A,3);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isa(M, 'single')); assert(~isreal(M)); assert(isequal(M, complex(single([2 3 4]), single([1 1 1]))));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink {
            label: "mean",
            target: BuiltinDocumentationLinkTarget::Builtin("mean"),
        },
        BuiltinDocumentationLink {
            label: "movsum",
            target: BuiltinDocumentationLinkTarget::Builtin("movsum"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/math/reduction/moving.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/moving.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::math::reduction::moving::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Native borrowed complex actions and publication (qualification pending)",
                location: "builtins::math::reduction::moving::complex::tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
