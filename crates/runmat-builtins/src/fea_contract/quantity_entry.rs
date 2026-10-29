//! Shared contract facts for dimensioned FEA values (`fea.Displacement.meters`,
//! `fea.Temperature.celsius`, `fea.Position.millimeters`, ...). Each
//! constructor validates one host numeric input, converts it once to canonical
//! SI binary64 storage, and returns an immutable value object.

use super::constructor_inference::{object_fact, resident};
use super::descriptors::ERRORS;
use crate::{
    catalog::{argument_error, finish_fixed},
    BuiltinCatalogEntry, BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_types::{
    CallInference, CallRequest, InferenceDiagnostic, NumericDomain, StaticClassIdentity, ValueFact,
    ValueKindFact,
};

pub const QUANTITY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real scalar in the unit named by the constructor.",
}];
pub const QUANTITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "quantity",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Immutable dimensioned FEA value in canonical SI storage.",
}];
pub const VECTOR_QUANTITY_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "components",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real three-component vector in the unit named by the constructor.",
}];
pub const VECTOR_QUANTITY_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "quantity",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Immutable three-component dimensioned FEA value in canonical meter storage.",
}];

pub const fn quantity_descriptor(
    signatures: &'static [BuiltinSignatureDescriptor],
) -> BuiltinDescriptor {
    BuiltinDescriptor {
        signatures,
        output_mode: BuiltinOutputMode::Fixed,
        completion_policy: BuiltinCompletionPolicy::Public,
        errors: &ERRORS,
    }
}

pub const QUANTITY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "value",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The exact scalar is validated before one explicit conversion to canonical binary64 SI storage; sufficiently wide integers can round.",
    }];
pub const QUANTITY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "dimensioned scalar value",
        inputs: &QUANTITY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Unit conversion and canonical storage are binary64 operations; resident values are rejected without provider execution.",
    }];
pub const VECTOR_QUANTITY_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "components",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "A three-component real integer vector is shape-checked before one explicit conversion to canonical binary64 meter storage; sufficiently wide integers can round.",
    }];
pub const VECTOR_QUANTITY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "three-component dimensioned vector",
        inputs: &VECTOR_QUANTITY_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Unit conversion and canonical vector storage are binary64 operations; resident values are rejected without provider execution.",
    }];

/// Admissible range of the canonical SI value, mirroring runtime validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantityRange {
    /// Any finite real value.
    Finite,
    /// `value >= 0`.
    NonNegative,
    /// `value > 0`.
    Positive,
    /// `0 <= value <= 1`.
    UnitInterval,
}

impl QuantityRange {
    pub fn admits(self, value: f64) -> bool {
        value.is_finite()
            && match self {
                Self::Finite => true,
                Self::NonNegative => value >= 0.0,
                Self::Positive => value > 0.0,
                Self::UnitInterval => (0.0..=1.0).contains(&value),
            }
    }
}

/// One scalar quantity constructor: result class, unit conversion
/// (`canonical = value * scale + offset`), and canonical range.
#[derive(Clone, Copy, Debug)]
pub struct ScalarQuantity {
    pub class: StaticClassIdentity,
    pub scale: f64,
    pub offset: f64,
    pub range: QuantityRange,
}

fn possible_real_numeric(fact: &ValueFact) -> bool {
    !resident(fact)
        && match &fact.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Numeric(number) => number.domain == NumericDomain::Real,
            _ => false,
        }
}

fn arity(request: &CallRequest, diagnostics: &mut Vec<InferenceDiagnostic>) {
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FEA-ARITY",
            "dimensioned value constructors take one numeric argument",
            request.arguments.len(),
        ));
    }
}

pub fn infer_scalar(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    quantity: ScalarQuantity,
) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics);
    if let Some(value) = request.arguments.first() {
        if !possible_real_numeric(value)
            || matches!(value.shape.element_count(), Some(count) if count != 1)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-QUANTITY",
                "the value must be a real numeric scalar on the host",
                0,
            ));
        }
        if let Some(literal) = request.literals.numeric_at(0) {
            if !quantity
                .range
                .admits(literal * quantity.scale + quantity.offset)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FEA-QUANTITY-RANGE",
                    "the value is outside the admissible range of this physical quantity",
                    0,
                ));
            }
        }
    }
    finish_fixed(entry, request, object_fact(quantity.class), diagnostics)
}

pub fn infer_vector(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: StaticClassIdentity,
) -> CallInference {
    let mut diagnostics = Vec::new();
    arity(request, &mut diagnostics);
    if let Some(value) = request.arguments.first() {
        if !possible_real_numeric(value)
            || matches!(value.shape.element_count(), Some(count) if count != 3)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEA-QUANTITY",
                "the components must be a real numeric vector with three elements on the host",
                0,
            ));
        }
    }
    finish_fixed(entry, request, object_fact(class), diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fea_contract::catalog_entry::{self, ConstructorEntryDefinition};
    use crate::{BuiltinCatalogProvenance, BuiltinDistributedPolicy, BuiltinDocumentation};
    use runmat_types::{
        DynamicReason, LiteralContext, LiteralValue, NumericClass, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact,
    };

    const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
        label: "value = fea.Temperature.celsius(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
    const DESCRIPTOR: BuiltinDescriptor = quantity_descriptor(&SIGNATURES);
    const CELSIUS: ScalarQuantity = ScalarQuantity {
        class: StaticClassIdentity::new("fea.Temperature"),
        scale: 1.0,
        offset: 273.15,
        range: QuantityRange::NonNegative,
    };

    fn entry() -> BuiltinCatalogEntry {
        catalog_entry::entry(ConstructorEntryDefinition {
            provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
            name: "fea.Temperature.celsius",
            documentation: BuiltinDocumentation::EMPTY,
            descriptor: &DESCRIPTOR,
            inference: |request, entry| infer_scalar(request, entry, CELSIUS),
            integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
            integer_audit: None,
            filesystem_read: false,
            distributed: BuiltinDistributedPolicy::Unsupported,
        })
    }

    fn request(arguments: Vec<ValueFact>, literals: Vec<LiteralValue>) -> CallRequest {
        CallRequest {
            arguments,
            literals: LiteralContext::new(literals),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        }
    }

    fn real(class: NumericClass) -> ValueFact {
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class,
            domain: NumericDomain::Real,
        }))
    }

    fn codes(inference: &CallInference) -> Vec<&str> {
        inference
            .diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect()
    }

    #[test]
    fn scalar_quantities_check_kind_and_converted_literal_range() {
        let entry = entry();
        let valid = infer_scalar(
            &request(
                vec![real(NumericClass::Int32)],
                vec![LiteralValue::Number(20.0)],
            ),
            &entry,
            CELSIUS,
        );
        assert!(valid.diagnostics.is_empty(), "{:?}", valid.diagnostics);
        match &valid.outputs[0].kind {
            ValueKindFact::Object(object) => assert!(object
                .runtime_class
                .as_ref()
                .is_some_and(|class| class.is(StaticClassIdentity::new("fea.Temperature")))),
            other => panic!("expected object, got {other:?}"),
        }
        let below_zero = infer_scalar(
            &request(
                vec![real(NumericClass::Double)],
                vec![LiteralValue::Number(-300.0)],
            ),
            &entry,
            CELSIUS,
        );
        assert!(codes(&below_zero).contains(&"RM-CATALOG-FEA-QUANTITY-RANGE"));
        let logical = infer_scalar(
            &request(
                vec![ValueFact::scalar(ValueKindFact::Logical)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
            CELSIUS,
        );
        assert!(codes(&logical).contains(&"RM-CATALOG-FEA-QUANTITY"));
        let arity = infer_scalar(&request(vec![], vec![]), &entry, CELSIUS);
        assert!(codes(&arity).contains(&"RM-CATALOG-FEA-ARITY"));
        let unknown = infer_scalar(
            &request(
                vec![ValueFact::unknown(DynamicReason::RuntimeValue)],
                vec![LiteralValue::Unknown],
            ),
            &entry,
            CELSIUS,
        );
        assert!(unknown.diagnostics.is_empty());
    }

    #[test]
    fn vector_quantities_require_three_components() {
        let entry = entry();
        let class = StaticClassIdentity::new("fea.Position");
        let mut row = real(NumericClass::Double);
        row.shape = ShapeFact::from(vec![Some(1), Some(3)]);
        assert!(infer_vector(
            &request(vec![row.clone()], vec![LiteralValue::Unknown]),
            &entry,
            class
        )
        .diagnostics
        .is_empty());
        row.shape = ShapeFact::from(vec![Some(1), Some(2)]);
        let short = infer_vector(
            &request(vec![row], vec![LiteralValue::Unknown]),
            &entry,
            class,
        );
        assert!(codes(&short).contains(&"RM-CATALOG-FEA-QUANTITY"));
    }

    #[test]
    fn ranges_match_runtime_rules() {
        assert!(QuantityRange::NonNegative.admits(0.0));
        assert!(!QuantityRange::Positive.admits(0.0));
        assert!(QuantityRange::UnitInterval.admits(1.0));
        assert!(!QuantityRange::UnitInterval.admits(1.5));
        assert!(!QuantityRange::Finite.admits(f64::INFINITY));
    }
}
