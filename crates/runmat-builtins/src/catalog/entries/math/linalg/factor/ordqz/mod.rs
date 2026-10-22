use crate::*;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, StorageFact, ValueFact, ValueKindFact,
};

const INPUTS: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "AA",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Triangular or real quasitriangular first QZ factor.",
    },
    BuiltinParamDescriptor {
        name: "BB",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Upper triangular second QZ factor.",
    },
    BuiltinParamDescriptor {
        name: "Q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left transform from the same QZ factorization.",
    },
    BuiltinParamDescriptor {
        name: "Z",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right transform from the same QZ factorization.",
    },
    BuiltinParamDescriptor {
        name: "selector",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Logical selection, lhp/rhp/udi/udo keyword, or positive-integer cluster vector.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "AAS",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reordered first generalized Schur factor.",
    },
    BuiltinParamDescriptor {
        name: "BBS",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Reordered second generalized Schur factor.",
    },
    BuiltinParamDescriptor {
        name: "QS",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Updated left orthogonal or unitary transform.",
    },
    BuiltinParamDescriptor {
        name: "ZS",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Updated right orthogonal or unitary transform.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[AAS,BBS,QS,ZS] = ordqz(AA,BB,Q,Z,selector)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];
const ERRORS: [BuiltinErrorDescriptor; 2] = [
    BuiltinErrorDescriptor {
        code: "RM.ORDQZ.INVALID_INPUT",
        identifier: Some("RunMat:ordqz:InvalidInput"),
        when: "Factors, selector, class, shape, or placement are invalid.",
        message: "ordqz: invalid generalized Schur form or selector",
    },
    BuiltinErrorDescriptor {
        code: "RM.ORDQZ.ILL_CONDITIONED",
        identifier: Some("RunMat:ordqz:IllConditioned"),
        when: "Adjacent generalized Schur blocks cannot be safely separated or swapped.",
        message: "ordqz: block exchange is ill-conditioned",
    },
];
pub const ORDQZ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    use crate::catalog::entries::math::linalg::factor::inference::{
        dimensions, extent, matrix_shape, schur_operand,
    };
    use crate::catalog::inference::{argument_error, finish_fixed_outputs};
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 5 {
        diagnostics.push(argument_error(
            "RM.ORDQZ.INVALID_INPUT",
            "ordqz expects AA, BB, Q, Z, and a selector",
            0,
        ));
    }
    for index in 0..request.arguments.len().min(4) {
        diagnostics.extend(schur_operand(
            &request.arguments[index],
            index,
            "RM.ORDQZ.INVALID_INPUT",
        ));
    }
    let input = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if let Some((Some(rows), Some(columns))) = dimensions(&input) {
        for index in 1..request.arguments.len().min(4) {
            if dimensions(&request.arguments[index])
                .is_some_and(|dims| dims == (Some(rows), Some(columns)))
            {
                continue;
            }
            if dimensions(&request.arguments[index]).is_some_and(|(other_rows, other_columns)| {
                other_rows.is_some_and(|value| value != rows)
                    || other_columns.is_some_and(|value| value != columns)
            }) {
                diagnostics.push(argument_error(
                    "RM.ORDQZ.INVALID_INPUT",
                    "ordqz factors must have the same order",
                    index,
                ));
            }
        }
    }
    let facts = request.arguments.iter().take(4).collect::<Vec<_>>();
    let class = if facts.len() == 4 {
        let numeric = facts
            .iter()
            .map(|fact| match &fact.kind {
                ValueKindFact::Numeric(value) => Some(*value),
                _ => None,
            })
            .collect::<Option<Vec<_>>>();
        numeric
            .filter(|values| {
                values
                    .iter()
                    .all(|value| matches!(value.class, NumericClass::Single | NumericClass::Double))
            })
            .map(|values| {
                let class = if values
                    .iter()
                    .any(|value| value.class == NumericClass::Double)
                {
                    NumericClass::Double
                } else {
                    NumericClass::Single
                };
                let domain = if values
                    .iter()
                    .any(|value| value.domain == NumericDomain::Complex)
                {
                    NumericDomain::Complex
                } else {
                    NumericDomain::Real
                };
                NumericFact { class, domain }
            })
    } else {
        None
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = matrix_shape(extent(&input), extent(&input));
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    if let Some(fact) = class {
        output.kind = ValueKindFact::Numeric(fact);
    }
    finish_fixed_outputs(entry, request, vec![output; 4], diagnostics)
}

const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection {
    heading: "Retained generalized Schur blocks",
    paragraphs: &[
        "ordqz reorders retained QZ factors with adjacent two-sided block exchanges. It does not form B\\A or re-factor the full pencil. QS*A*ZS=AAS and QS*B*ZS=BBS remain the transform authority.",
        "A real 2-by-2 conjugate pair is indivisible: if either logical member is selected, both are selected; mismatched numeric cluster labels use the larger label. Numeric clusters appear in descending index order, while logical and region selections move selected blocks first. Region keywords are lhp (real part <0), rhp (>=0), udi (modulus <1), and udo (>=1). Nonzero/zero homogeneous pairs are infinite; 0/0 pairs cannot be classified by a region keyword.",
        "The four factor inputs are caller-supplied generalized Schur factors. Shape, class, finite entries and triangular/quasitriangular structure are checked, but a cubic full-pair unitary/reconstruction check is not repeated for each local exchange. Unseparable or uncertifiable blocks raise an operational error before publication.",
        "This portable sequential implementation has no provider `ordqz` hook or cooperative CPU-budget checkpoint; resident factors are rejected before host computation.",
    ],
}];
const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "ordqz-infinite-leading",
    title: "Move an infinite generalized eigenvalue first",
    program: "A = [2 0; 0 3]; B = [1 0; 0 0]; [AA,BB,Q,Z] = qz(A,B); [AAS,BBS,QS,ZS] = ordqz(AA,BB,Q,Z,[false;true]);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(norm(QS*A*ZS-AAS)<1e-8); assert(norm(QS*B*ZS-BBS)<1e-8);",
    },
}];
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ordqz"),
    slug: Some("ordqz"),
    summary: "Reorder retained generalized Schur blocks.",
    description: "Reorder real or complex QZ factors by logical, region, or cluster selection without refactoring the whole pencil.",
    keywords: &["ordqz", "qz", "generalized Schur", "eigenvalue clusters"],
    related: &["qz", "ordeig", "ordschur"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[],
    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Real adjacent QZ block exchange",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/factor/qz/real_reorder.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Real conjugate-pair and infinite-block reconstruction source fixture",
            location: "builtins::math::linalg::factor::qz::tests::real_ordqz_moves_infinite_block_past_conjugate_pair_without_complexifying",
        }],
        notes: &["Source-only candidate; actual numerical examples, native/WASM, resource and cancellation qualification remain open."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub const ORDQZ_CATALOG_ENTRY: BuiltinCatalogEntry =
    super::entry::owned_host_cpu_factor_entry(super::entry::FactorEntrySpec {
        name: "ordqz",
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        documentation: DOCUMENTATION,
        descriptor: &ORDQZ_DESCRIPTOR,
        inference: IdentityInferenceRule::new(infer),
        extensions: &[],
        integer_capabilities: &[],
        residency: BuiltinResidencyPolicy::Host,
        maturity: BuiltinContractMaturity::Incomplete,
    });

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ORDQZ_CATALOG_ENTRY];
