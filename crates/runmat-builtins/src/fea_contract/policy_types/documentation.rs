//! Example and evidence layout shared by typed FEA policy documentation.

use crate::*;

/// A portable, executable example whose assertions check the constructed value.
pub const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

pub const fn source(label: &'static str, url: &'static str) -> BuiltinDocumentationLink {
    BuiltinDocumentationLink {
        label,
        target: BuiltinDocumentationLinkTarget::Source(url),
    }
}

pub const fn page(label: &'static str, route: &'static str) -> BuiltinDocumentationLink {
    BuiltinDocumentationLink {
        label,
        target: BuiltinDocumentationLinkTarget::Documentation(route),
    }
}

pub const fn unit_test(label: &'static str, location: &'static str) -> BuiltinEvidenceReference {
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label,
        location,
    }
}

pub const fn section(
    heading: &'static str,
    paragraphs: &'static [&'static str],
) -> BuiltinDocumentationSection {
    BuiltinDocumentationSection {
        heading,
        paragraphs,
    }
}

pub const fn faq(question: &'static str, answer: &'static str) -> BuiltinDocumentationFaq {
    BuiltinDocumentationFaq { question, answer }
}

pub const STUDY_CONTRACTS: BuiltinDocumentationLink = page(
    "Procedure sequences, observations, and accuracy",
    "/docs/simulation/study-contracts",
);
