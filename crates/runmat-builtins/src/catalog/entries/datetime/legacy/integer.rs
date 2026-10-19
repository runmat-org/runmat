use crate::*;

const EXTENSION_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "legacy numeric date input", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Typed integers use authoritative native storage, then cross the checked floating serial-date boundary in RunMat mode.",
}];
const DOCUMENTED_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "integer-valued date or calendar component", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Documented integer-valued input is checked in native storage before the host calendar calculation.",
}];
const BETWEEN_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "numeric serial-date operand", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The numeric serial-date isbetween convenience form is gated in MATLAB mode; its result is logical.",
}];

macro_rules! conversion {
    ($name:ident, $form:literal, $inputs:expr, $class:expr, $domain:expr) => {
        pub(super) const $name: [BuiltinIntegerCapabilityDescriptor; 1] = [BuiltinIntegerCapabilityDescriptor {
            form: $form,
            inputs: $inputs,
            computation_domain: $domain,
            output_class: $class,
            overflow: BuiltinIntegerOverflowRule::Error,
            backend: BuiltinIntegerBackendRule::GatherFallback,
            overload: BuiltinIntegerOverloadKind::Multiple,
            notes: "Typed values are range-checked at the host calendar boundary; accepted resident inputs require the RunMat extension and gather through their provider owner.",
        }];
    };
}

conversion!(
    DATENUM,
    "DateNumber = datenum(integer_components_or_serial)",
    &EXTENSION_INPUT,
    BuiltinIntegerOutputClassRule::Double,
    BuiltinIntegerComputationDomain::Structural
);
conversion!(
    DATESTR,
    "DateString = datestr(integer_serial)",
    &EXTENSION_INPUT,
    BuiltinIntegerOutputClassRule::FunctionSpecific,
    BuiltinIntegerComputationDomain::FloatingPoint
);
conversion!(
    DATEVEC,
    "DateVector = datevec(integer_serial)",
    &EXTENSION_INPUT,
    BuiltinIntegerOutputClassRule::Double,
    BuiltinIntegerComputationDomain::FloatingPoint
);
conversion!(
    EOMDAY,
    "D = eomday(integer_year,integer_month)",
    &DOCUMENTED_INPUT,
    BuiltinIntegerOutputClassRule::Double,
    BuiltinIntegerComputationDomain::Structural
);
conversion!(
    ETIME,
    "seconds = etime(integer_date_vectors)",
    &EXTENSION_INPUT,
    BuiltinIntegerOutputClassRule::Double,
    BuiltinIntegerComputationDomain::FloatingPoint
);
conversion!(
    ISBETWEEN,
    "tf = isbetween(integer_serial_dates)",
    &BETWEEN_INPUT,
    BuiltinIntegerOutputClassRule::Logical,
    BuiltinIntegerComputationDomain::Predicate
);
conversion!(
    WEEKDAY,
    "DayNumber = weekday(integer_serial)",
    &DOCUMENTED_INPUT,
    BuiltinIntegerOutputClassRule::Double,
    BuiltinIntegerComputationDomain::Structural
);
