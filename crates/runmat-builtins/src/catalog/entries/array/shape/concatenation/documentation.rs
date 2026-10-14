use crate::*;

const HORZ_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "horzcat implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/horzcat.rs"),
};
const VERT_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "vertcat implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/shape/vertcat.rs"),
};
const HORZ_TESTS: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "horzcat host, provider and like-prototype tests",
        location: "builtins::array::shape::horzcat::tests",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "horzcat gpuArray roundtrip",
        location: "builtins::array::shape::horzcat::tests::horzcat_gpu_roundtrip",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "horzcat host inputs with gpuArray prototype",
        location: "builtins::array::shape::horzcat::tests::horzcat_like_gpu_from_host_inputs",
    },
];
const VERT_TESTS: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "vertcat host, provider and like-prototype tests",
        location: "builtins::array::shape::vertcat::tests",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "vertcat gpuArray roundtrip",
        location: "builtins::array::shape::vertcat::tests::vertcat_gpu_roundtrip",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "vertcat host inputs with gpuArray prototype",
        location: "builtins::array::shape::vertcat::tests::vertcat_like_gpu_from_host_inputs",
    },
];
macro_rules! example {
    ($id:literal, $title:literal, $program:literal, $display:expr, $assertions:literal, $harness:expr) => {
        BuiltinExample {
            id: $id,
            title: $title,
            program: $program,
            display_output: $display,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: $harness,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: $assertions,
            },
        }
    };
}

mod horzcat;
mod vertcat;

pub(super) use horzcat::HORZ_DOC;
pub(super) use vertcat::VERT_DOC;
