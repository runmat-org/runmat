use crate::*;

const STRUCTURAL_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "ind", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Positive one-based indices are decoded exactly and range-checked without a floating-point intermediary." },
    BuiltinIntegerInputCapability { name: "sz", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "Positive output dimensions are decoded exactly and checked against platform and materialization bounds." },
];
const DEFAULT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "data", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Integer scalar or vector data is grouped from native storage." },
    BuiltinIntegerInputCapability { name: "fillval", classes: &[], availability: BuiltinIntegerInputAvailability::Rejected, scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable, notes: "Default integer-data summation returns double, so a typed integer fill is not accepted for that form." },
];
const CALLBACK_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The callback receives native-class integer column vectors.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "An integer fill is valid when its class matches the callback scalar result.",
    },
];
const SPARSE_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "data",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Sparse output requires double data.",
    },
    BuiltinIntegerInputCapability {
        name: "fillval",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Sparse output requires an omitted or double-zero fill.",
    },
    BuiltinIntegerInputCapability {
        name: "issparse",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The sparse selector accepts logical or numeric scalar 0 or 1.",
    },
];

pub const ACCUMARRAY_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor { form: "B = accumarray(integer_ind,data,integer_sz)", inputs: &STRUCTURAL_INPUTS, computation_domain: BuiltinIntegerComputationDomain::Structural, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GatherFallback, overload: BuiltinIntegerOverloadKind::StructuralParameter, notes: "Indices and dimensions remain exact through bounds checks and column-major linearization." },
    BuiltinIntegerCapabilityDescriptor { form: "B = accumarray(ind,integer_data,sz,[])", inputs: &DEFAULT_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::Double, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "Default summation returns double; resident integer data is rejected by the compatible GPU-array form." },
    BuiltinIntegerCapabilityDescriptor { form: "B = accumarray(ind,integer_data,sz,fun,integer_fillval)", inputs: &CALLBACK_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific, output_class: BuiltinIntegerOutputClassRule::FunctionSpecific, overflow: BuiltinIntegerOverflowRule::FunctionSpecific, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::SameSizeOrScalar, notes: "The callback result determines output class and the fill must match it." },
    BuiltinIntegerCapabilityDescriptor { form: "B = accumarray(ind,integer_data,sz,fun,integer_fillval,true)", inputs: &SPARSE_INPUTS, computation_domain: BuiltinIntegerComputationDomain::FloatingPoint, output_class: BuiltinIntegerOutputClassRule::NotApplicable, overflow: BuiltinIntegerOverflowRule::NotApplicable, backend: BuiltinIntegerBackendRule::HostOnly, overload: BuiltinIntegerOverloadKind::Multiple, notes: "Sparse output is double-only and rejects typed integer data and fills." },
];
