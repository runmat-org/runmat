use crate::*;

const DATE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Datetime, serial date number, or supported date text.",
};
const VECTOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "DateVector",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One date per row, with three or six numeric components.",
};
const TEXT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "DateString",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Supported date text in a string or character array.",
};
const FORMAT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "format",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Legacy date-format text (mm is month; MM is minute).",
};
const FORMAT_CODE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "formatCode",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Host real double scalar integer from -1 through 31; -1 selects the whole-input default.",
};
const YEAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Calendar year; scalar values expand to the other components' shape.",
};
const MONTH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Calendar month; scalar values expand to the other components' shape.",
};
const DAY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Calendar day; scalar values expand to the other components' shape.",
};
const SERIAL_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "DateNumber",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double serial date values.",
}];
const DOUBLE_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "out",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double result.",
}];
const CHAR_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "DateString",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "One formatted character row per date.",
}];
const LOGICAL_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host logical comparison result.",
}];
const COMPONENT_OUT: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Year column.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Month column.",
    },
    BuiltinParamDescriptor {
        name: "D",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Day column.",
    },
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Hour column.",
    },
    BuiltinParamDescriptor {
        name: "MI",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minute column.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second column, possibly fractional.",
    },
];
const DATEVEC_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "DateVector",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "An N-by-6 double matrix, one date per row.",
}];
const WEEKDAY_OUT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "DayNumber",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Host double day number, Sunday = 1.",
    },
    BuiltinParamDescriptor {
        name: "DayName",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "English short or long character rows.",
    },
];
const WEEKDAY_NUMBER_OUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "DayNumber",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host double day number, Sunday = 1.",
}];
const DAY_FORM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "DayForm",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("short"),
    description: "'short' or 'long' English day names; other locales are not implemented.",
};
const COMPONENTS_THREE: [BuiltinParamDescriptor; 3] = [YEAR, MONTH, DAY];
const COMPONENTS_SIX: [BuiltinParamDescriptor; 6] = [
    YEAR,
    MONTH,
    DAY,
    BuiltinParamDescriptor {
        name: "H",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Hour component.",
    },
    BuiltinParamDescriptor {
        name: "MI",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Minute component.",
    },
    BuiltinParamDescriptor {
        name: "S",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second component, possibly fractional.",
    },
];
const DATE_FORMAT: [BuiltinParamDescriptor; 2] = [DATE, FORMAT];
const DATE_FORMAT_CODE: [BuiltinParamDescriptor; 2] = [DATE, FORMAT_CODE];
const DATE_FORM: [BuiltinParamDescriptor; 2] = [DATE, DAY_FORM];
const DATE_ONLY: [BuiltinParamDescriptor; 1] = [DATE];
const VECTOR_ONLY: [BuiltinParamDescriptor; 1] = [VECTOR];
const TEXT_ONLY: [BuiltinParamDescriptor; 1] = [TEXT];
const EOM_INPUTS: [BuiltinParamDescriptor; 2] = [YEAR, MONTH];
const ETIME_INPUTS: [BuiltinParamDescriptor; 2] = [VECTOR, VECTOR];
const BETWEEN_INPUTS: [BuiltinParamDescriptor; 3] = [DATE, DATE, DATE];

pub(super) const DATENUM: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "DateNumber = datenum()",
            inputs: &[],
            outputs: &SERIAL_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "DateNumber = datenum(t)",
            inputs: &DATE_ONLY,
            outputs: &SERIAL_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "DateNumber = datenum(DateVector)",
            inputs: &VECTOR_ONLY,
            outputs: &SERIAL_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "DateNumber = datenum(DateString)",
            inputs: &TEXT_ONLY,
            outputs: &SERIAL_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "DateNumber = datenum(Y,M,D)",
            inputs: &COMPONENTS_THREE,
            outputs: &SERIAL_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "DateNumber = datenum(Y,M,D,H,MI,S)",
            inputs: &COMPONENTS_SIX,
            outputs: &SERIAL_OUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(super) const DATESTR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "DateString = datestr(D)",
            inputs: &DATE_ONLY,
            outputs: &CHAR_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "DateString = datestr(D,format)",
            inputs: &DATE_FORMAT,
            outputs: &CHAR_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "DateString = datestr(D,formatCode)",
            inputs: &DATE_FORMAT_CODE,
            outputs: &CHAR_OUT,
        },
    ],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(super) const DATEVEC: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "DateVector = datevec(D)",
            inputs: &DATE_ONLY,
            outputs: &DATEVEC_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "[Y,M,D,H,MI,S] = datevec(D)",
            inputs: &DATE_ONLY,
            outputs: &COMPONENT_OUT,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(super) const EOMDAY: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "D = eomday(Y,M)",
        inputs: &EOM_INPUTS,
        outputs: &DOUBLE_OUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(super) const ETIME: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "seconds = etime(t2,t1)",
        inputs: &ETIME_INPUTS,
        outputs: &DOUBLE_OUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(super) const ISBETWEEN: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "tf = isbetween(t,lower,upper)",
        inputs: &BETWEEN_INPUTS,
        outputs: &LOGICAL_OUT,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(super) const WEEKDAY: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[
        BuiltinSignatureDescriptor {
            label: "DayNumber = weekday(D)",
            inputs: &DATE_ONLY,
            outputs: &WEEKDAY_NUMBER_OUT,
        },
        BuiltinSignatureDescriptor {
            label: "[DayNumber,DayName] = weekday(D,DayForm)",
            inputs: &DATE_FORM,
            outputs: &WEEKDAY_OUT,
        },
    ],
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
