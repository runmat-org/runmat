//! Parameter facts shared by every typed physical-signal constructor.

use crate::fea_contract::policy_types::descriptor::{output, required};
use crate::{BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType};

pub const SIGNAL_OUTPUT: [BuiltinParamDescriptor; 1] = [output(
    "signal",
    "Immutable typed physical-signal object of the constructor's class.",
)];

pub const SCALAR_CONSTANT_INPUTS: [BuiltinParamDescriptor; 2] = [
    required("id", BuiltinParamType::StringScalar, "Nonempty signal identifier referenced by excitations, initial conditions, and constraints."),
    required("value", BuiltinParamType::NumericScalar, "Finite real scalar in the canonical SI unit named by the constructor."),
];

pub const VECTOR_CONSTANT_INPUTS: [BuiltinParamDescriptor; 2] = [
    required("id", BuiltinParamType::StringScalar, "Nonempty signal identifier referenced by excitations, initial conditions, and constraints."),
    required("components", BuiltinParamType::NumericArray, "Finite 1-by-3 or 3-by-1 real vector of global-frame components in the canonical SI unit named by the constructor."),
];

pub const TIME_SAMPLE_INPUTS: [BuiltinParamDescriptor; 5] = [
    required("id", BuiltinParamType::StringScalar, "Nonempty signal identifier."),
    required("times", BuiltinParamType::NumericArray, "Finite, strictly increasing time samples in seconds; a row or column vector with at least two entries."),
    required("values", BuiltinParamType::NumericArray, "Finite samples in the class's canonical SI unit: one scalar per time for scalar signals, or an N-by-3 or 3-by-N matrix of global components for vector signals."),
    required("interpolation", BuiltinParamType::Any, "Typed fea.SignalInterpolation policy used between samples."),
    required("extrapolation", BuiltinParamType::Any, "Typed fea.SignalExtrapolation policy used outside the first and last time."),
];

pub const FUNCTION_INPUTS: [BuiltinParamDescriptor; 5] = [
    required("id", BuiltinParamType::StringScalar, "Nonempty signal identifier."),
    required("callable", BuiltinParamType::Callable, "Function handle or closure bound to the executing RunMat program; it is frozen with its captures."),
    required("domain", BuiltinParamType::Any, "Typed fea.SignalDomain value naming the independent variable."),
    required("sampling", BuiltinParamType::Any, "Typed fea.SignalSampling value naming the abscissae at which the callable is evaluated."),
    BuiltinParamDescriptor {
        name: "Name, Value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Optional 'Arguments' (cell array of extra callable arguments), 'Interpolation' (default linear), and 'Extrapolation' (default reject).",
    },
];
