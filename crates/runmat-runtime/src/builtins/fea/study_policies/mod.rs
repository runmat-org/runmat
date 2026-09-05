pub(super) mod accuracy;
pub(super) mod observation;
pub(super) mod sequence;

pub(super) fn ensure_study_policy_classes_registered() {
    sequence::ensure_classes_registered();
    observation::ensure_classes_registered();
    accuracy::ensure_classes_registered();
}
