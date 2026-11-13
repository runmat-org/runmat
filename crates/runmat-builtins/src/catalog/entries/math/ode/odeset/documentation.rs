use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("odeset"), slug: Some("odeset"),
    summary: "Create and merge options for RunMat ODE solvers.",
    description: "Build a scalar options structure for ode45, ode23, and ode15s without changing the option payloads.",
    keywords: &["odeset", "ODE", "tolerance", "step size", "options"],
    related: &["ode45", "ode23", "ode15s"],
    sections: &[
        BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
            "Use odeset(name,value,...) to create options, odeset(oldopts,name,value,...) to update them, or odeset(oldopts,newopts) to merge structures. An empty numeric oldopts starts with defaults.",
            "Names are case-insensitive and allow unambiguous abbreviations. Full names take precedence. Later name/value pairs replace earlier values.",
            "When merging structures, only nonempty new values replace old values. An explicit name/value pair with [] resets that option. Unspecified fields are returned as [] so the selected solver supplies its defaults.",
            "With no inputs and one output, returns the supported fields with empty values. With no inputs and no outputs, lists the supported options and defaults.",
        ]},
        BuiltinDocumentationSection { heading: "Options and limits", paragraphs: &[
            "Supports RelTol, AbsTol, InitialStep, MaxStep, and Refine. MaxSteps is a separately gated RunMat resource-control extension. Values are retained without numeric conversion; the selected solver validates their class, shape, bounds, and method-specific support.",
            "The constructor does not add solver capabilities. Unsupported names, including Events, OutputFcn, Jacobian, and Mass, raise an error instead of being stored and ignored. This is not the complete MATLAB ODE/PDE options surface.",
            "ODE solvers accept scalar AbsTol; vector AbsTol is unsupported. The one-output ode45, ode23, and ode15s forms return the y trajectory, not a solution structure.",
            "Native integer payloads and explicit gpuArray payloads are separately gated RunMat extensions. No option payload is gathered by this constructor; solver admission owns resident execution policy.",
        ]},
    ],
    examples: &[
        BuiltinExample {
            id: "tolerances", title: "Create and update tolerances",
            program: "opts = odeset('RelTol',1e-6,'AbsTol',1e-8); opts = odeset(opts,'MaxStep',0.1); [t,y] = ode45(@(t,y) -y,[0 1],1,opts);",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(opts.RelTol == 1e-6); assert(opts.AbsTol == 1e-8); assert(opts.MaxStep == 0.1); assert(isempty(opts.InitialStep)); assert(size(t,1)==size(y,1)); assert(abs(y(end)-exp(-1))<1e-4);" },
        },
        BuiltinExample {
            id: "merge", title: "Merge only specified new settings",
            program: "old = odeset('RelTol',1e-5,'AbsTol',1e-7); new = odeset('MaxStep',0.2); opts = odeset(old,new);",
            display_output: None, compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(opts.RelTol == 1e-5); assert(opts.AbsTol == 1e-7); assert(opts.MaxStep == 0.2);" },
        },
    ],
    example_exemption: None, faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "MATLAB odeset", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/odeset.html") },
        BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/ode/odeset.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest, label: "Options construction and merge tests",
            location: "builtins::math::ode::odeset::tests",
        }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
