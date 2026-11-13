use crate::{
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness, BuiltinExampleVerification,
};

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const REFERENCE_LINKS:
    &[BuiltinDocumentationLink] = &[
    builtin("plus"),
    builtin("minus"),
    builtin("times"),
    builtin("rdivide"),
    builtin("ldivide"),
    builtin("power"),
    builtin("mrdivide"),
    builtin("mldivide"),
    builtin("gpuArray"),
    builtin("gather"),
    builtin("abs"),
    builtin("angle"),
    builtin("conj"),
    builtin("double"),
    builtin("single"),
    builtin("exp"),
    builtin("expm1"),
    builtin("factorial"),
    builtin("gamma"),
    builtin("hypot"),
    builtin("imag"),
    builtin("log"),
    builtin("log10"),
    builtin("log1p"),
    builtin("log2"),
    builtin("pow2"),
    builtin("real"),
    builtin("sign"),
    builtin("sqrt"),
];

const fn builtin(name: &'static str) -> BuiltinDocumentationLink {
    BuiltinDocumentationLink {
        label: name,
        target: BuiltinDocumentationLinkTarget::Builtin(name),
    }
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const fn matlab_example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    example(
        id,
        title,
        program,
        display_output,
        assertions,
        BuiltinExampleCompatibility::Matlab,
        BuiltinExampleHarness::Portable,
    )
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const fn matlab_wgpu_example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    example(
        id,
        title,
        program,
        display_output,
        assertions,
        BuiltinExampleCompatibility::Matlab,
        BuiltinExampleHarness::Wgpu,
    )
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const fn runmat_wgpu_example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: &'static str,
    assertions: &'static str,
) -> BuiltinExample {
    example(
        id,
        title,
        program,
        display_output,
        assertions,
        BuiltinExampleCompatibility::RunMat,
        BuiltinExampleHarness::Wgpu,
    )
}

const fn example(
    id: &'static str,
    title: &'static str,
    program: &'static str,
    display_output: &'static str,
    assertions: &'static str,
    compatibility: BuiltinExampleCompatibility,
    harness: BuiltinExampleHarness,
) -> BuiltinExample {
    BuiltinExample {
        id,
        title,
        program,
        display_output: Some(display_output),
        compatibility,
        harness,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: assertions },
    }
}

pub(in crate::catalog::entries::math::elementwise::binary_arithmetic) const fn faq(
    question: &'static str,
    answer: &'static str,
) -> BuiltinDocumentationFaq {
    BuiltinDocumentationFaq { question, answer }
}
