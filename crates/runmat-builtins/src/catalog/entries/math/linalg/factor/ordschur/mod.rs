use crate::*;
use runmat_types::{CallInference, CallRequest, DynamicReason, ValueFact};

const INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "U",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Schur vectors.",
    },
    BuiltinParamDescriptor {
        name: "T",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real quasitriangular or complex triangular Schur form.",
    },
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Logical selection, lhp/rhp/udi/udo keyword, or positive-integer clusters.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "US",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reordered orthogonal/unitary Schur vectors.",
    },
    BuiltinParamDescriptor {
        name: "TS",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reordered Schur form.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[US,TS] = ordschur(U,T,selector)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    BuiltinErrorDescriptor {
        code: "RM.ORDSCHUR.INVALID_INPUT",
        identifier: Some("RunMat:ordschur:InvalidInput"),
        when: "Factor shapes, classes, structure, or selector are invalid.",
        message: "ordschur: invalid Schur factors or selector",
    },
    BuiltinErrorDescriptor {
        code: "RM.ORDSCHUR.ILL_CONDITIONED",
        identifier: Some("RunMat:ordschur:IllConditioned"),
        when: "Selected adjacent blocks cannot be separated by a stable invariant subspace.",
        message: "ordschur: Schur blocks cannot be separated",
    },
    BuiltinErrorDescriptor {
        code: "RM.ORDSCHUR.INVALID_OUTPUTS",
        identifier: Some("RunMat:ordschur:InvalidOutputs"),
        when: "More than two outputs are requested.",
        message: "ordschur: at most two outputs are supported",
    },
];
pub const ORDSCHUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
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
    if request.arguments.len() != 3 {
        diagnostics.push(argument_error(
            "RM.ORDSCHUR.INVALID_INPUT",
            "U, T, and a selector are required",
            0,
        ));
    }
    for (index, operand) in request.arguments.iter().take(2).enumerate() {
        diagnostics.extend(schur_operand(operand, index, "RM.ORDSCHUR.INVALID_INPUT"));
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
                "RM.ORDSCHUR.INVALID_INPUT",
                "Schur factors must have equal shape",
                1,
            ));
        }
    }
    let input = request
        .arguments
        .get(1)
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let mut output = factor_output(
        &input,
        matrix_shape(extent(&input), extent(&input)),
        false,
        false,
        true,
    );
    if let [u, t, ..] = request.arguments.as_slice() {
        match (&u.kind, &t.kind) {
            (
                runmat_types::ValueKindFact::Numeric(u_numeric),
                runmat_types::ValueKindFact::Numeric(t_numeric),
            ) if u_numeric.class == t_numeric.class => {
                let mut numeric = *t_numeric;
                if u_numeric.domain == runmat_types::NumericDomain::Complex {
                    numeric.domain = runmat_types::NumericDomain::Complex;
                }
                output.kind = runmat_types::ValueKindFact::Numeric(numeric);
            }
            _ => output.kind = runmat_types::ValueKindFact::Unknown,
        }
    }
    finish_fixed_outputs(entry, request, vec![output.clone(), output], diagnostics)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection { heading: "Block-preserving order", paragraphs: &[
    "[US,TS] = ordschur(U,T,selector) moves selected eigenvalue blocks to the upper-left while preserving U*T*U' = US*TS*US'.",
    "A logical selector selects either member of a real conjugate pair as a whole. Keywords 'lhp', 'rhp', 'udi', and 'udo' select spectral regions. Positive-integer cluster labels sort in descending order; a real pair with different labels joins the larger label.",
    "A local block exchange that cannot certify its invariant subspace fails before publishing any factors. No generalized QZ reordering is implied.",
    "U is expected to be the caller's orthogonal or unitary Schur factor; ordschur checks finite structural inputs but does not refactor U or recompute U'*U.",
] }];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "ordschur-negative",
    title: "Move negative eigenvalues first",
    program: "A = diag([2 -3 1]); [U,T] = schur(A); [US,TS] = ordschur(U,T,'lhp');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(abs(TS(1,1)+3) < 1e-10); assert(norm(A-US*TS*US') < 1e-10);",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("ordschur"), slug: Some("ordschur"),
    summary: "Reorder selected eigenvalue blocks in an ordinary Schur factorization.",
    description: "Apply stable, adjacent, block-preserving transformations to retained ordinary Schur factors.",
    keywords: &["ordschur", "schur", "eigenvalue ordering"], related: &["schur", "ordeig", "rsf2csf"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: &[], links: &[], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Schur block exchange", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/schur/reorder.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "reorder residual and order", location: "builtins::math::linalg::factor::schur::tests::adjacent_exchange_preserves_real_schur_reconstruction_and_order" }], notes: &[] },
    introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

pub const ORDSCHUR_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::entry::host_factor_entry(super::entry::FactorEntrySpec {
        name: "ordschur",
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: DOCUMENTATION,
        descriptor: &ORDSCHUR_DESCRIPTOR,
        inference: IdentityInferenceRule::new(infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: BuiltinResidencyPolicy::Host,
        maturity: BuiltinContractMaturity::Complete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ORDSCHUR_CATALOG_ENTRY];
