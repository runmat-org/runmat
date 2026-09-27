use crate::*;
const INTEGER_GROUP_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "group",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "Integer labels are exact and must use the same class as grouphat and Order.",
}];
const INTEGER_GROUPHAT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "grouphat",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Integer predicted labels are exact and must use the same class as group and Order.",
    }];
const INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Order",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
    notes: "Integer Order labels are deduplicated and matched without floating conversion.",
}];
pub const CONFUSIONMAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor { form: "C = confusionmat(integer_group, grouphat)", inputs: &INTEGER_GROUP_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "RunMat mode groups and sorts all eight integer classes exactly; C contains double counts." },
    BuiltinIntegerCapabilityDescriptor { form: "C = confusionmat(group, integer_grouphat)", inputs: &INTEGER_GROUPHAT_INPUT, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::Multiple, notes: "This role is independently gated before gather; integer label classes must match." },
    BuiltinIntegerCapabilityDescriptor { form: "[C, order] = confusionmat(integer_group, integer_grouphat, 'Order', integer_order)", inputs: &INTEGER_ORDER_INPUT, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "C contains double counts; the optional order output retains the exact integer label class and values, including values above flintmax." },
];
