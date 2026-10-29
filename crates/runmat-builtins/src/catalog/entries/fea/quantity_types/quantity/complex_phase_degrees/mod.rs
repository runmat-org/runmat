use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    identities,
    quantity_entry::{
        infer_scalar, quantity_descriptor, QuantityRange, ScalarQuantity, QUANTITY_INPUTS,
        QUANTITY_INTEGER_CAPABILITIES, QUANTITY_OUTPUTS,
    },
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

const QUANTITY: ScalarQuantity = ScalarQuantity {
    class: identities::FEA_COMPLEX_PHASE_CLASS,
    scale: std::f64::consts::PI / 180.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const COMPLEX_PHASE_DEGREES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.ComplexPhase.degrees(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const COMPLEX_PHASE_DEGREES_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&COMPLEX_PHASE_DEGREES_SIGNATURES);
const COMPLEX_PHASE_DEGREES_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "complex-phase-degrees",
    title: "A 45 degree phase shift",
    program: "value = fea.ComplexPhase.degrees(45);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.ComplexPhase'));\nassert(abs(value.radians - 0.7853981633974483) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ComplexPhase.degrees",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ComplexPhase.degrees",
        summary: "Create a typed ComplexPhase value from degrees.",
        description: "`value = fea.ComplexPhase.degrees(x)` returns a `fea.ComplexPhase` holding a phase shift between periodic images, with `x` given in degrees and stored in radians (rad).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "ComplexPhase",
                "degrees",
            ],
        related: &[
                "fea.ComplexPhase.radians",
                "fea.AcousticConstraint.phaseShiftedPeriodic",
                "fea.MagneticConstraint.phaseShiftedPeriodic",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a phase shift between periodic images in degrees. It is converted to radians (rad) by multiplying by pi/180, and the result is an immutable `fea.ComplexPhase` whose `radians` property holds the canonical value in radians (rad).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a phase shift between periodic images, such as `fea.AcousticConstraint.phaseShiftedPeriodic`, `fea.MagneticConstraint.phaseShiftedPeriodic`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: COMPLEX_PHASE_DEGREES_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &COMPLEX_PHASE_DEGREES_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
