pub(crate) mod accuracy;
pub(crate) mod observation;
pub(crate) mod sequence;

pub(super) fn ensure_study_policy_classes_registered() {
    sequence::ensure_classes_registered();
    observation::ensure_classes_registered();
    accuracy::ensure_classes_registered();
}
