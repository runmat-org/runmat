use crate::*;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericDomain, ValueFact, ValueKindFact,
};

const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real orthogonal Schur vectors.",
    },
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real quasitriangular Schur form.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Unew",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex unitary Schur vectors.",
    },
    BuiltinParamDescriptor {
        name: "Tnew",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Complex triangular Schur form.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[Unew,Tnew] = rsf2csf(U,T)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.RSF2CSF.INVALID_INPUT",
        identifier: Some("RunMat:rsf2csf:InvalidInput"),
        when: "Factors are not matching real square Schur matrices.",
        message: "rsf2csf: expected matching real Schur factors",
    },
    BuiltinErrorDescriptor {
        code: "RM.RSF2CSF.NUMERICAL",
        identifier: Some("RunMat:rsf2csf:Numerical"),
        when: "A 2-by-2 block cannot be triangularized within the bounded iteration.",
        message: "rsf2csf: complex block conversion failed",
    },
];
pub const RSF2CSF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    use crate::catalog::entries::math::linalg::factor::inference::{
        dimensions, extent, factor_output, matrix_shape, schur_operand,
    };
    use crate::catalog::inference::{argument_error, finish_fixed_outputs};
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM.RSF2CSF.INVALID_INPUT",
            "two Schur factors are required",
            0,
        ));
    }
    for (index, operand) in request.arguments.iter().take(2).enumerate() {
        diagnostics.extend(schur_operand(operand, index, "RM.RSF2CSF.INVALID_INPUT"));
        if matches!(&operand.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex)
        {
            diagnostics.push(argument_error(
                "RM.RSF2CSF.INVALID_INPUT",
                "real Schur factors are required",
                index,
            ));
        }
    }
    if let [u, t, ..] = request.arguments.as_slice() {
        if dimensions(u)
            .zip(dimensions(t))
            .is_some_and(|((ur, uc), (tr, tc))| {
                ur.zip(tr).is_some_and(|(ur, tr)| ur != tr)
                    || uc.zip(tc).is_some_and(|(uc, tc)| uc != tc)
            })
        {
            diagnostics.push(argument_error(
                "RM.RSF2CSF.INVALID_INPUT",
                "Schur factors must have equal shape",
                1,
            ));
        }
    }
    let input = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let mut output = factor_output(
        &input,
        matrix_shape(extent(&input), extent(&input)),
        false,
        false,
        true,
    );
    if let ValueKindFact::Numeric(mut numeric) = output.kind.clone() {
        numeric.domain = NumericDomain::Complex;
        output.kind = ValueKindFact::Numeric(numeric);
    }
    finish_fixed_outputs(entry, request, vec![output.clone(), output], diagnostics)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection { heading: "Real-to-complex Schur conversion", paragraphs: &[
    "[Unew,Tnew] = rsf2csf(U,T) converts retained real Schur factors to complex Schur factors without recomputing the source matrix.",
    "The transformation acts on each exact structural 2-by-2 block, accumulates its unitary rotation in Unew, and keeps Unew*Tnew*Unew' equal to U*T*U'. Inputs must be real, finite, dense, and of one floating class.",
    "U is a caller-supplied orthogonal Schur factor; the structural and class checks do not recompute U'*U before this local transformation.",
] }];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "rsf2csf-rotation",
    title: "Triangularize a conjugate-pair block",
    program: "A = [0 -1; 1 0]; [U,T] = schur(A); [Uc,Tc] = rsf2csf(U,T);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(norm(A-Uc*Tc*Uc') < 1e-10); assert(abs(Tc(2,1)) < 1e-10);",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("rsf2csf"), slug: Some("rsf2csf"),
    summary: "Convert real Schur factors to complex triangular Schur factors.",
    description: "Apply unitary transformations to a retained real Schur form and its Schur vectors.",
    keywords: &["rsf2csf", "schur", "complex"], related: &["schur", "ordschur", "ordeig"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Schur complexification", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/schur/form.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "complexification residual", location: "builtins::math::linalg::factor::schur::tests::complexification_preserves_factorization" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

pub const RSF2CSF_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::entry::host_factor_entry(super::entry::FactorEntrySpec {
        name: "rsf2csf",
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: DOCUMENTATION,
        descriptor: &RSF2CSF_DESCRIPTOR,
        inference: IdentityInferenceRule::new(infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: BuiltinResidencyPolicy::Host,
        maturity: BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&RSF2CSF_CATALOG_ENTRY];
