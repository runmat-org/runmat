use crate::{
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

pub(super) const fn builtin(name: &'static str) -> BuiltinDocumentationLink {
    BuiltinDocumentationLink {
        label: name,
        target: BuiltinDocumentationLinkTarget::Builtin(name),
    }
}

pub(super) const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: &'static str,
    assertions: &'static str,
    harness: BuiltinExampleHarness,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: Some(display_output),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

pub(super) const fn expected_error_example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: &'static str,
    identifier: &'static str,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: Some(display_output),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::ExpectedError { identifier },
    }
}

pub(super) const fn faq(question: &'static str, answer: &'static str) -> BuiltinDocumentationFaq {
    BuiltinDocumentationFaq { question, answer }
}
