use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ArrayInferenceRule {
    Accumulation(AccumulationInferenceRule),
    Binning(BinningInferenceRule),
    Combinatorics(CombinatoricsInferenceRule),
    Creation(ArrayCreationInferenceRule),
    Grouping(GroupingInferenceRule),
    Introspection(ArrayIntrospectionInferenceRule),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AccumulationInferenceRule {
    Indexed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinningInferenceRule {
    Discretize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CombinatoricsInferenceRule {
    CartesianProduct,
    Permutations,
    SelectionCombinations,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GroupingInferenceRule {
    Counts,
    GroupedApply,
    IndexLabels,
    SortedGroups,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ArrayCreationInferenceRule {
    Full,
    Zeros,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ArrayIntrospectionInferenceRule {
    ShapePredicate(ShapePredicate),
    ShapeQuery(ShapeQuery),
    ShapeScalarQuery(ShapeScalarQuery),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapeQuery {
    Size,
    ElementCount,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapePredicate {
    Empty,
    Scalar,
    Vector,
    Matrix,
    Row,
    Column,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapeScalarQuery {
    Length,
    Rank,
    Height,
    Width,
}
