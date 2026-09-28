use super::capabilities::{CALENDAR_DURATION_CLASS, DATETIME_CLASS};
use super::*;

pub(super) type Broadcast3 = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

pub(super) static DATETIME_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(DATETIME_CLASS);
pub(super) static CALENDAR_DURATION_CLASS_REGISTERED: crate::class_registry::ClassRegistration =
    crate::class_registry::ClassRegistration::new(CALENDAR_DURATION_CLASS);

// Legacy datetime-family method descriptors share the constructor's canonical
// error declarations until their own catalog entries are migrated.
pub(super) use runmat_builtins::{
    DATETIME_ERRORS, DATETIME_ERROR_INVALID_ARGUMENT, DATETIME_ERROR_INVALID_INPUT,
};

pub(super) const OUT_DATETIME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Datetime object result.",
}];
pub(super) const OUT_NUMERIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric scalar/tensor result.",
}];
pub(super) const OUT_ANY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Method result.",
}];
pub(super) const DATETIME_SINGLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Datetime input.",
}];
pub(super) const DATETIME_BINARY_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Left datetime operand.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Right datetime/numeric/duration operand.",
    },
];
pub(super) const DATESHIFT_INPUTS: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "t",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime input.",
    },
    BuiltinParamDescriptor {
        name: "boundary",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Shift form: 'start', 'end', or 'dayofweek'.",
    },
    BuiltinParamDescriptor {
        name: "unit",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Calendar/time unit.",
    },
    BuiltinParamDescriptor {
        name: "rule",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Optional current/next/previous/nearest or integer occurrence rule.",
    },
];
pub(super) const DATETIME_SUBSREF_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime receiver object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
];
pub(super) const DATETIME_SUBSASGN_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "obj",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Datetime receiver object.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Standard substruct-compatible indexing path.",
    },
    BuiltinParamDescriptor {
        name: "rhs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Assigned value.",
    },
];

pub(super) const DATETIME_YEAR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = year(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = year(t, yearTypeOrFormat)",
        inputs: DATETIME_HOUR_SIGNATURES[1].inputs,
        outputs: &OUT_NUMERIC,
    },
];
pub(super) const DATETIME_MONTH_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = month(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = month(t, monthTypeOrFormat)",
        inputs: DATETIME_HOUR_SIGNATURES[1].inputs,
        outputs: &OUT_ANY,
    },
];
pub(super) const DATETIME_DAY_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = day(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = day(t, dayType)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "t",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Datetime, legacy serial date number, or date text.",
            },
            BuiltinParamDescriptor {
                name: "dayType",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "dayofmonth, dayofweek, iso-dayofweek, dayofyear, name, or shortname.",
            },
        ],
        outputs: &OUT_ANY,
    },
];
pub(super) const DATETIME_HOUR_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = hour(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = hour(t, F)",
        inputs: &[
            BuiltinParamDescriptor {
                name: "t",
                ty: BuiltinParamType::Any,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Legacy serial date number or date text.",
            },
            BuiltinParamDescriptor {
                name: "F",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: None,
                description: "Legacy datestr input format.",
            },
        ],
        outputs: &OUT_NUMERIC,
    },
];
pub(super) const DATETIME_MINUTE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "X = minute(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    },
    BuiltinSignatureDescriptor {
        label: "X = minute(t, F)",
        inputs: DATETIME_HOUR_SIGNATURES[1].inputs,
        outputs: &OUT_NUMERIC,
    },
];
pub(super) const DATETIME_SECOND_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "X = second(t)",
        inputs: &DATETIME_SINGLE_INPUT,
        outputs: &OUT_NUMERIC,
    }];
pub(super) const DATETIME_SUBSREF_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = datetime.subsref(obj, S)",
        inputs: &DATETIME_SUBSREF_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(super) const DATETIME_SUBSASGN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = datetime.subsasgn(obj, S, rhs)",
        inputs: &DATETIME_SUBSASGN_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(super) const DATETIME_BINARY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "out = datetime.op(lhs, rhs)",
        inputs: &DATETIME_BINARY_INPUTS,
        outputs: &OUT_ANY,
    }];
pub(super) const DATESHIFT_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "t2 = dateshift(t, boundary, unit)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t2 = dateshift(t, boundary, unit, rule)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
    BuiltinSignatureDescriptor {
        label: "t2 = dateshift(t, \"dayofweek\", weekday)",
        inputs: &DATESHIFT_INPUTS,
        outputs: &OUT_DATETIME,
    },
];
