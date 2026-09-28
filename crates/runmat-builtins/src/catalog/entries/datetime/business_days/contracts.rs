use crate::*;

const START: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "StartDate",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar or array of datetime, text, or serial dates.",
};
const END: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "EndDate",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Scalar or compatible array of end dates.",
};
const HOLIDAY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Holiday",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("RunMat market calendar"),
    description: "Date vector of nontrading days; an empty array selects the default calendar.",
};
const WEEKEND: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Weekend",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Optional,
    default: Some("[1 0 0 0 0 0 1]"),
    description: "Seven Sunday-first weekend indicators.",
};
const BASIS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Basis",
    ty: BuiltinParamType::IntegerScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0"),
    description: "Currently supported day-count bases: 0 (actual) and 1 (30/360 SIA).",
};
const PERIOD: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "bdmode",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: Some("1 (daily)"),
    description: "Daily, weekly, monthly, quarterly, semiannual, or annual period.",
};
const YEAR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Year",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Calendar year or array of years.",
};
const MONTH: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Month",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Month number or compatible array of months.",
};
const DATE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Date",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Datetime, text, or serial date input.",
};
const NUMBER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "NumDays",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Double day counts or serial dates.",
}];
const DATES: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Dates",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Column vector of dates.",
}];
const LOGICAL: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Busday",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical mask matching the input dates.",
}];

const BUSDAYS_INPUTS: [BuiltinParamDescriptor; 4] = [START, END, PERIOD, HOLIDAY];
const BUSDAYS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Dates = busdays(StartDate, EndDate, bdmode, Holiday)",
    inputs: &BUSDAYS_INPUTS,
    outputs: &DATES,
}];
pub(super) const BUSDAYS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BUSDAYS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const DAYS252BUS_INPUTS: [BuiltinParamDescriptor; 3] = [START, END, HOLIDAY];
const DAYS252BUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "NumDays = days252bus(StartDate, EndDate, Holiday)",
    inputs: &DAYS252BUS_INPUTS,
    outputs: &NUMBER,
}];
pub(super) const DAYS252BUS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DAYS252BUS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const DAYSDIF_INPUTS: [BuiltinParamDescriptor; 3] = [START, END, BASIS];
const DAYSDIF_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "NumDays = daysdif(StartDate, EndDate, Basis)",
    inputs: &DAYSDIF_INPUTS,
    outputs: &NUMBER,
}];
pub(super) const DAYSDIF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DAYSDIF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const MONTH_OPTIONS: [BuiltinParamDescriptor; 5] = [
    YEAR,
    MONTH,
    HOLIDAY,
    WEEKEND,
    BuiltinParamDescriptor {
        name: "outputType",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("datenum or Holiday's datetime class"),
        description: "datenum or datetime output selection.",
    },
];
const FIRST_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Date = fbusdate(Year, Month, Holiday, Weekend, outputType)",
    inputs: &MONTH_OPTIONS,
    outputs: &DATES,
}];
const LAST_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Date = lbusdate(Year, Month, Holiday, Weekend, outputType)",
    inputs: &MONTH_OPTIONS,
    outputs: &DATES,
}];
pub(super) const FBUSDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FIRST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
pub(super) const LBUSDATE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LAST_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const HOLIDAYS_INPUTS: [BuiltinParamDescriptor; 3] = [
    START,
    END,
    BuiltinParamDescriptor {
        name: "AltHolidays",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Alternate dates to return within the requested inclusive range.",
    },
];
const HOLIDAYS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Dates = holidays()",
        inputs: &[],
        outputs: &DATES,
    },
    BuiltinSignatureDescriptor {
        label: "Dates = holidays(Year) [RunMat mode]",
        inputs: &[YEAR],
        outputs: &DATES,
    },
    BuiltinSignatureDescriptor {
        label: "Dates = holidays(StartDate, EndDate, AltHolidays)",
        inputs: &HOLIDAYS_INPUTS,
        outputs: &DATES,
    },
];
pub(super) const HOLIDAYS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &HOLIDAYS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

const ISBUSDAY_INPUTS: [BuiltinParamDescriptor; 3] = [DATE, HOLIDAY, WEEKEND];
const ISBUSDAY_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Busday = isbusday(Date, Holiday, Weekend)",
    inputs: &ISBUSDAY_INPUTS,
    outputs: &LOGICAL,
}];
pub(super) const ISBUSDAY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ISBUSDAY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
