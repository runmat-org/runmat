use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Coordinate conventions and outputs",
        paragraphs: &[
            "`[x,y] = pol2cart(theta,rho)` converts polar coordinates to Cartesian coordinates. `[x,y,z] = pol2cart(theta,rho,z)` converts cylindrical coordinates. Angles are radians measured counterclockwise from the positive x-axis; rho is radial distance and z is elevation.",
            "The formulas are `x = rho .* cos(theta)` and `y = rho .* sin(theta)`. All inputs participate in MATLAB-style implicit expansion, including z when only x and y are requested. The returned elevation is expanded without a trigonometric conversion. Empty and N-D arrays are supported.",
            "The two-input form supports zero, one, or two requested outputs; the three-input form supports up to three. A single assignment or implicit output returns x. A discarded-output call still validates and evaluates the transform.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Numeric classes",
        paragraphs: &[
            "Inputs must be real dense single or double scalars or arrays. The trigonometric step retains theta's floating precision; the x/y multiplication returns single when theta or rho is single, otherwise double. The z output retains the elevation's floating class independently.",
            "RunMat mode independently permits all eight fixed-width integer classes for theta, rho, and z, provided every value is exactly representable as double. Integer values are validated before conversion and cross into floating computation once. MATLAB compatibility mode rejects these extensions. Logical, character, string, sparse, cell, struct, object, and complex inputs are not accepted.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Provider execution",
        paragraphs: &[
            "Homogeneous floating inputs can be placed automatically after the complete two- or three-input form is validated. All data operands, the expanded output shape, and the requested output count participate in placement. Mixed floating classes retain the host formula so that coordinate and elevation precision are not silently changed.",
            "Compatible real resident inputs can use their exact owning provider's sine, cosine, and broadcasted multiply operations. Borrowed input handles are preserved. Newly allocated intermediates and unrequested outputs are released on completion, failure, or cancellation.",
            "Only an unsupported provider operation selects host fallback. Other provider failures and invalid output metadata remain errors. Mixed physical floating classes or providers use the host formula. Unsupported paths can return host values, so residency is not guaranteed for every input form.",
            "This multiple-output transform terminates elementwise fusion rather than presenting a scalar fusion template. Distributed inputs use the shared materialization route.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "polar", title: "Convert polar coordinates",
        program: "theta = [0 pi/4 pi/2 pi];\nrho = [5 5 10 10];\n[x,y] = pol2cart(theta,rho)",
        display_output: Some("x = [5 3.5355 0 -10], y = [0 3.5355 10 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(x - [5 5/sqrt(2) 0 -10])) < 1e-12);\nassert(max(abs(y - [0 5/sqrt(2) 10 0])) < 1e-12);" },
    },
    BuiltinExample {
        id: "cylindrical", title: "Convert cylindrical coordinates",
        program: "theta = [0 pi/2]';\nrho = [1 2 3];\nz = 7;\n[x,y,z] = pol2cart(theta,rho,z)",
        display_output: Some("x = [1 2 3; 0 0 0], y = [0 0 0; 1 2 3], z = 7*ones(2,3)"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "ex = [1 2 3; 0 0 0]; ey = [0 0 0; 1 2 3];\nassert(max(abs(x(:)-ex(:))) < 1e-12);\nassert(max(abs(y(:)-ey(:))) < 1e-12);\nassert(isequal(z, 7*ones(2,3)));" },
    },
    BuiltinExample {
        id: "independent-single-elevation", title: "Expand single elevation into an N-D grid",
        program: "elevation = reshape(single([7 8]), [1 1 2]);\n[x,y,z] = pol2cart(single(0), [2 3], elevation);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(x,'single') && isa(y,'single') && isa(z,'single'));\nassert(isequal(size(x),[1 2 2]));\nassert(isequal(double(x(:)),[2;3;2;3]));\nassert(isequal(double(y(:)),zeros(4,1)));\nassert(isequal(double(z(:)),[7;7;8;8]));" },
    },
    BuiltinExample {
        id: "gpu-cylindrical", title: "Expand resident coordinates using elevation",
        program: "theta = gpuArray([0; pi/2]);\nrho = gpuArray(2);\nz = gpuArray([7 8 9]);\n[x,y,z] = pol2cart(theta,rho,z);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(x,'gpuArray') && isa(y,'gpuArray') && isa(z,'gpuArray'));\nx = gather(x); y = gather(y); z = gather(z);\nex = [2 2 2;0 0 0]; ey = [0 0 0;2 2 2];\nassert(isequal(size(x),[2 3]));\nassert(max(abs(x(:)-ex(:))) < 1e-6);\nassert(max(abs(y(:)-ey(:))) < 1e-6);\nassert(isequal(z,[7 8 9;7 8 9]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pol2cart"), slug: Some("pol2cart"),
    summary: "Transform polar or cylindrical coordinates to Cartesian coordinates.",
    description: "Convert real polar or cylindrical coordinate arrays with implicit expansion and requested-output handling.",
    keywords: &["pol2cart", "polar", "cylindrical", "cartesian", "coordinates", "trigonometry", "gpu"],
    related: &["atan2", "hypot", "cos", "sin"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What angle units does pol2cart use?", answer: "Theta is in radians, measured counterclockwise from the positive x-axis." },
        BuiltinDocumentationFaq { question: "Can I request only two outputs with three inputs?", answer: "Yes. [x,y] = pol2cart(theta,rho,z) includes z in expansion but returns only x and y." },
        BuiltinDocumentationFaq { question: "Does pol2cart accept complex inputs?", answer: "No. Inputs must be real." },
        BuiltinDocumentationFaq { question: "Does RunMat keep pol2cart on the GPU?", answer: "Compatible real inputs remain resident when their owning provider supports the required sine, cosine, and broadcasted multiply operations. Unsupported forms fall back to host evaluation." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "atan2", target: BuiltinDocumentationLinkTarget::Builtin("atan2") },
        BuiltinDocumentationLink { label: "hypot", target: BuiltinDocumentationLinkTarget::Builtin("hypot") },
        BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
        BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/pol2cart.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coordinate shapes, classes, requested outputs and provider lifetimes", location: "crates/runmat-runtime/src/builtins/math/trigonometry/pol2cart/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
