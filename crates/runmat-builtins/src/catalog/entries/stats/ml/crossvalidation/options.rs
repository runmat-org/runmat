//! Public cross-validation methods and options, decoded once at call boundaries.

fn normalized(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

/// The two public partition helpers retain distinct floating-count grammars.
/// Typed integer storage is decoded separately, without a binary64 detour.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartitionCountGrammar {
    CrossvalInd,
    CvPartition,
}

impl PartitionCountGrammar {
    pub fn positive_from_float(self, value: f64) -> Option<usize> {
        let integer = match self {
            Self::CrossvalInd => value.fract().abs() <= 1.0e-12,
            Self::CvPartition => value.fract() == 0.0,
        };
        (value.is_finite()
            && integer
            && value >= 1.0
            && value <= usize::MAX as f64
            && !(usize::BITS == 64 && value == usize::MAX as f64))
            .then_some(value as usize)
    }
}

/// The legacy index-and-mask API's supported methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossvalIndMethod {
    KFold,
    HoldOut,
    LeaveMOut,
    Resubstitution,
}

impl CrossvalIndMethod {
    pub fn parse(value: &str) -> Option<Self> {
        match normalized(value).as_str() {
            "kfold" => Some(Self::KFold),
            "holdout" => Some(Self::HoldOut),
            "leavemout" | "leavem" => Some(Self::LeaveMOut),
            "resubstitution" | "resub" => Some(Self::Resubstitution),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::KFold => "KFold",
            Self::HoldOut => "HoldOut",
            Self::LeaveMOut => "LeaveMOut",
            Self::Resubstitution => "Resubstitution",
        }
    }
}

/// The object constructor's supported methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvPartitionMethod {
    KFold,
    Holdout,
    Leaveout,
    Resubstitution,
    Custom,
}

impl CvPartitionMethod {
    pub fn parse(value: &str) -> Option<Self> {
        match normalized(value).as_str() {
            "kfold" => Some(Self::KFold),
            "holdout" => Some(Self::Holdout),
            "leaveout" => Some(Self::Leaveout),
            "resubstitution" => Some(Self::Resubstitution),
            "custompartition" => Some(Self::Custom),
            _ => None,
        }
    }

    /// Decode the object's Type property, whose spelling differs from the
    /// constructor token for custom partitions.
    pub fn from_property_name(value: &str) -> Option<Self> {
        let name = normalized(value);
        [
            Self::KFold,
            Self::Holdout,
            Self::Leaveout,
            Self::Resubstitution,
            Self::Custom,
        ]
        .into_iter()
        .find(|method| method.property_name() == name)
    }

    pub fn property_name(self) -> &'static str {
        match self {
            Self::KFold => "kfold",
            Self::Holdout => "holdout",
            Self::Leaveout => "leaveout",
            Self::Resubstitution => "resubstitution",
            Self::Custom => "custom",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CrossvalIndOption {
    Classes,
    Min,
}

impl CrossvalIndOption {
    pub fn parse(value: &str) -> Option<Self> {
        match normalized(value).as_str() {
            "classes" => Some(Self::Classes),
            "min" => Some(Self::Min),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvPartitionOption {
    Stratify,
}

impl CvPartitionOption {
    pub fn parse(value: &str) -> Option<Self> {
        match normalized(value).as_str() {
            "stratify" => Some(Self::Stratify),
            _ => None,
        }
    }
}

/// Boolean text aliases are accepted only at the public Stratify boundary.
/// Execution and inference retain the decoded bool thereafter.
pub fn parse_stratify_boolean_alias(value: &str) -> Option<bool> {
    if value.eq_ignore_ascii_case("true") {
        Some(true)
    } else if value.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

/// Numeric aliases are a RunMat extension. This tolerance is the public
/// scalar-alias rule, not a general comparison tolerance for label identity.
pub fn parse_stratify_numeric_boolean(value: f64) -> Option<bool> {
    if (value - 0.0).abs() <= 1.0e-12 {
        Some(false)
    } else if (value - 1.0).abs() <= 1.0e-12 {
        Some(true)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaskSelectionKeyword {
    All,
}

impl MaskSelectionKeyword {
    pub fn parse(value: &str) -> Option<Self> {
        (normalized(value) == "all").then_some(Self::All)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_grammars_keep_distinct_public_forms() {
        assert_eq!(
            CrossvalIndMethod::parse("Leave_M-Out"),
            Some(CrossvalIndMethod::LeaveMOut)
        );
        assert_eq!(
            CrossvalIndMethod::parse("resub"),
            Some(CrossvalIndMethod::Resubstitution)
        );
        assert_eq!(CrossvalIndMethod::parse("leaveout"), None);
        assert_eq!(
            CvPartitionMethod::parse("CustomPartition"),
            Some(CvPartitionMethod::Custom)
        );
        assert_eq!(
            CvPartitionMethod::parse("LeaveOut"),
            Some(CvPartitionMethod::Leaveout)
        );
        assert_eq!(CvPartitionMethod::parse("leavemout"), None);
        assert_eq!(CvPartitionMethod::parse("custom"), None);
        for method in [
            CvPartitionMethod::KFold,
            CvPartitionMethod::Holdout,
            CvPartitionMethod::Leaveout,
            CvPartitionMethod::Resubstitution,
            CvPartitionMethod::Custom,
        ] {
            assert_eq!(
                CvPartitionMethod::from_property_name(method.property_name()),
                Some(method)
            );
        }
    }

    #[test]
    fn option_and_mask_tokens_have_one_typed_decoder() {
        assert_eq!(
            CrossvalIndOption::parse("CLASSES"),
            Some(CrossvalIndOption::Classes)
        );
        assert_eq!(
            CrossvalIndOption::parse("Min"),
            Some(CrossvalIndOption::Min)
        );
        assert_eq!(
            CvPartitionOption::parse("Stratify"),
            Some(CvPartitionOption::Stratify)
        );
        assert_eq!(
            MaskSelectionKeyword::parse("all"),
            Some(MaskSelectionKeyword::All)
        );
        assert_eq!(MaskSelectionKeyword::parse("selected"), None);
    }

    #[test]
    fn numeric_count_grammars_share_their_runtime_and_static_authority() {
        let near_two = 2.0 + 5.0e-13;
        assert_eq!(
            PartitionCountGrammar::CrossvalInd.positive_from_float(near_two),
            Some(2)
        );
        assert_eq!(
            PartitionCountGrammar::CvPartition.positive_from_float(near_two),
            None
        );
        assert_eq!(
            PartitionCountGrammar::CvPartition.positive_from_float(2.0),
            Some(2)
        );
        assert_eq!(
            PartitionCountGrammar::CrossvalInd.positive_from_float(f64::NAN),
            None
        );
    }
}
