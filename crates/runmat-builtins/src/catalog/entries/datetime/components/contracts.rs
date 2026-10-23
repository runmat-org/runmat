use crate::*;

/// The two supported today output forms share one canonical option parser.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TodayOutputType {
    Datenum,
    Datetime,
}

impl TodayOutputType {
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("datenum") {
            Some(Self::Datenum)
        } else if value.eq_ignore_ascii_case("datetime") {
            Some(Self::Datetime)
        } else {
            None
        }
    }
}

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
pub const DATETIME_YEAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_YEAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_MONTH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_MONTH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_DAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_DAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_HOUR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_HOUR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
pub const DATETIME_MINUTE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DATETIME_MINUTE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};

macro_rules! current_descriptor {
    ($constant:ident, $label:literal, $ty:ident, $description:literal) => {
        pub const $constant: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &[BuiltinSignatureDescriptor {
                label: $label,
                inputs: &[],
                outputs: &[BuiltinParamDescriptor {
                    name: "out",
                    ty: BuiltinParamType::$ty,
                    arity: BuiltinParamArity::Required,
                    default: None,
                    description: $description,
                }],
            }],
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &DATETIME_ERRORS,
        };
    };
}
current_descriptor!(
    NOW_DESCRIPTOR,
    "n = now()",
    NumericScalar,
    "Scalar double local serial date number, including time of day."
);
const TODAY_OUTPUT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "date",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar double serial date by default, or scalar datetime for outputType='datetime'; both represent local midnight.",
}];
pub const TODAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "date = today()",
            inputs: &[],
            outputs: TODAY_OUTPUT,
        },
        BuiltinSignatureDescriptor {
            label: "date = today(outputType)",
            inputs: &[BuiltinParamDescriptor {
                name: "outputType",
                ty: BuiltinParamType::StringScalar,
                arity: BuiltinParamArity::Required,
                default: Some("datenum"),
                description: "datenum (default) or datetime; string scalar or character vector.",
            }],
            outputs: TODAY_OUTPUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DATETIME_ERRORS,
};
current_descriptor!(
    CLOCK_DESCRIPTOR,
    "c = clock()",
    NumericArray,
    "1-by-6 double local date vector [year month day hour minute second]."
);
