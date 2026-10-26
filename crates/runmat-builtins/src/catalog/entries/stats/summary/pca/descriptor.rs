use crate::*;
pub const PCA_ERROR_INVALID_ARGUMENT:BuiltinErrorDescriptor=BuiltinErrorDescriptor {
    code:"RM.PCA.INVALID_ARGUMENT",identifier:Some("RunMat:pca:InvalidArgument"),
    when:"Class, matrix geometry, retained finite observations/degrees of freedom, supported options, device intent or output count is invalid.",message:"pca: invalid argument",
};
const INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Real single/double matrix, observations by variables.",
    },
    BuiltinParamDescriptor {
        name: "name_value_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Algorithm, Centered, Economy, NumComponents and Rows options.",
    },
];
const OUTPUTS: [BuiltinParamDescriptor; 6] = [
    BuiltinParamDescriptor {
        name: "coeff",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Principal directions, variables by returned score components.",
    },
    BuiltinParamDescriptor {
        name: "score",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Observation scores with excluded rows reinserted as NaN.",
    },
    BuiltinParamDescriptor {
        name: "latent",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Component variances, column vector, independent of score truncation.",
    },
    BuiltinParamDescriptor {
        name: "tsquared",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Hotelling T-squared over all degrees-of-freedom-admissible PCs, before NumComponents truncation.",
    },
    BuiltinParamDescriptor {
        name: "explained",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Percentage variance explained, with the same extent as latent.",
    },
    BuiltinParamDescriptor {
        name: "mu",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Variable means, or zeros when Centered is off.",
    },
];
pub const PCA_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &[BuiltinSignatureDescriptor {
        label: "[coeff,score,latent,tsquared,explained,mu] = pca(X,Name,Value,...)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    }],
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[PCA_ERROR_INVALID_ARGUMENT],
};
