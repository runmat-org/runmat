use crate::*;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact, ValueKindFact};

const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "T",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Ordinary real quasitriangular or complex triangular Schur form.",
}];
const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "E",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Eigenvalues in Schur-block order.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "E = ordeig(T)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.ORDEIG.INVALID_INPUT",
        identifier: Some("RunMat:ordeig:InvalidInput"),
        when: "T is not a finite Schur-form matrix.",
        message: "ordeig: expected an ordinary Schur form",
    },
    BuiltinErrorDescriptor {
        code: "RM.ORDEIG.QZ_PENDING",
        identifier: Some("RunMat:ordeig:QzPending"),
        when: "Generalized two-matrix form is requested.",
        message: "ordeig(AA,BB) requires generalized Schur support",
    },
];
pub const ORDEIG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    use crate::catalog::entries::math::linalg::factor::inference::{
        extent, factor_output, matrix_shape, schur_operand,
    };
    use crate::catalog::inference::{argument_error, finish_fixed_outputs};
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM.ORDEIG.INVALID_INPUT",
            "a Schur form is required",
            0,
        ));
    } else if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM.ORDEIG.QZ_PENDING",
            "generalized ordeig(AA,BB) is not yet implemented",
            1,
        ));
    }
    let input = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if !request.arguments.is_empty() {
        diagnostics.extend(schur_operand(&input, 0, "RM.ORDEIG.INVALID_INPUT"));
    }
    let mut output = factor_output(
        &input,
        matrix_shape(extent(&input), Some(1)),
        false,
        true,
        true,
    );
    output.kind = ValueKindFact::Unknown;
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection { heading: "Ordered Schur spectrum", paragraphs: &[
    "E = ordeig(T) reads 1-by-1 and 2-by-2 diagonal blocks of an ordinary Schur form in their existing order. A nonzero real subdiagonal marks an indivisible 2-by-2 block, even when it is tiny.",
    "Only the one-input ordinary Schur form is implemented. Generalized ordeig(AA,BB) awaits a retained QZ factorization and is rejected, not approximated through B\\A.",
] }];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "ordeig-ordered",
    title: "Read a Schur spectrum without sorting",
    program: "T = diag([3 -1 2]); e = ordeig(T);",
    display_output: Some("e = [3; -1; 2]"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(e,[3;-1;2]));",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("ordeig"), slug: Some("ordeig"),
    summary: "Extract eigenvalues from ordinary Schur blocks in factor order.",
    description: "Read the spectrum of a retained ordinary Schur form without reordering it; generalized QZ form remains pending.",
    keywords: &["ordeig", "schur", "eigenvalues"], related: &["schur", "ordschur", "eig"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Schur block spectrum", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/schur/form.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "tiny complex-pair block", location: "builtins::math::linalg::factor::schur::tests::tiny_real_schur_subdiagonal_is_structural_not_discarded" }], notes: &["Generalized ordeig(AA,BB) is not yet implemented."] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};

pub const ORDEIG_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::entry::host_factor_entry(super::entry::FactorEntrySpec {
        name: "ordeig",
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: DOCUMENTATION,
        descriptor: &ORDEIG_DESCRIPTOR,
        inference: IdentityInferenceRule::new(infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: BuiltinResidencyPolicy::Host,
        maturity: BuiltinContractMaturity::Incomplete,
    });
