#include "preview_face_owners.hxx"

#include <TDF_LabelSequence.hxx>
#include <TDataStd_Name.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <XCAFDoc_ShapeTool.hxx>

#include <algorithm>
#include <cmath>
#include <limits>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

bool has_name(const TDF_Label& label) {
  Handle(TDataStd_Name) name;
  return !label.IsNull() && label.FindAttribute(TDataStd_Name::GetID(), name) && !name.IsNull() &&
         name->Get().Length() > 0;
}

double placement_distance(const gp_Trsf& lhs, const gp_Trsf& rhs) {
  double distance = 0.0;
  for (Standard_Integer row = 1; row <= 3; ++row) {
    for (Standard_Integer column = 1; column <= 4; ++column) {
      distance = std::max(distance, std::abs(lhs.Value(row, column) - rhs.Value(row, column)));
    }
  }
  return distance;
}

} // namespace

PreviewFaceOwners PreviewFaceOwners::build(const Handle(XCAFDoc_ShapeTool)& shape_tool) {
  PreviewFaceOwners owners;
  if (shape_tool.IsNull()) {
    return owners;
  }
  TDF_LabelSequence free_shapes;
  shape_tool->GetFreeShapes(free_shapes);
  for (Standard_Integer index = 1; index <= free_shapes.Length(); ++index) {
    owners.add_occurrence(free_shapes.Value(index), {}, gp_Trsf(), 0);
  }
  return owners;
}

void PreviewFaceOwners::add_occurrence(const TDF_Label& label,
                                       std::vector<TDF_Label> occurrences,
                                       const gp_Trsf& parent,
                                       std::size_t depth) {
  if (label.IsNull() || depth > 256) {
    return;
  }
  TDF_Label definition = label;
  TDF_Label referred;
  gp_Trsf placement = parent;
  if (XCAFDoc_ShapeTool::IsReference(label) &&
      XCAFDoc_ShapeTool::GetReferredShape(label, referred) && !referred.IsNull()) {
    definition = referred;
    // A component places its product; the product's own shape keeps any location it carries.
    placement.Multiply(XCAFDoc_ShapeTool::GetLocation(label).Transformation());
  }
  occurrences.push_back(label);
  if (XCAFDoc_ShapeTool::IsAssembly(definition)) {
    TDF_LabelSequence components;
    if (XCAFDoc_ShapeTool::GetComponents(definition, components, Standard_False)) {
      for (Standard_Integer index = 1; index <= components.Length(); ++index) {
        add_occurrence(components.Value(index), occurrences, placement, depth + 1);
      }
    }
    return;
  }
  const TopoDS_Shape shape = XCAFDoc_ShapeTool::GetShape(definition);
  if (shape.IsNull()) {
    return;
  }
  retained_.push_back(shape);
  const std::size_t leaf = leaves_.size();
  leaves_.push_back(Leaf{definition, std::move(occurrences), placement});
  for (TopExp_Explorer faces(shape, TopAbs_FACE); faces.More(); faces.Next()) {
    gp_Trsf face_placement = placement;
    face_placement.Multiply(faces.Current().Location().Transformation());
    uses_[faces.Current().TShape().get()].push_back(Use{leaf, face_placement});
  }
  // Named faces of this product: a label that holds faces (and no solid) names each face.
  TDF_LabelSequence subshapes;
  if (!XCAFDoc_ShapeTool::GetSubShapes(definition, subshapes)) {
    return;
  }
  for (Standard_Integer index = 1; index <= subshapes.Length(); ++index) {
    const TDF_Label& subshape = subshapes.Value(index);
    if (!has_name(subshape)) {
      continue;
    }
    const TopoDS_Shape named = XCAFDoc_ShapeTool::GetShape(subshape);
    if (named.IsNull() || TopExp_Explorer(named, TopAbs_SOLID).More()) {
      continue;
    }
    retained_.push_back(named);
    for (TopExp_Explorer faces(named, TopAbs_FACE); faces.More(); faces.Next()) {
      named_faces_.emplace(faces.Current().TShape().get(), subshape);
    }
  }
}

bool PreviewFaceOwners::resolve(const TopoDS_Face& face, PreviewFaceOwner& owner) const {
  const auto found = uses_.find(face.TShape().get());
  if (found == uses_.end() || found->second.empty()) {
    return false;
  }
  // One product placed several times shares its faces' kernel shapes; the face's own placement
  // picks the occurrence.
  const gp_Trsf placement = face.Location().Transformation();
  const Use* best = nullptr;
  double best_distance = std::numeric_limits<double>::infinity();
  for (const Use& use : found->second) {
    const double distance = placement_distance(use.placement, placement);
    if (distance < best_distance) {
      best_distance = distance;
      best = &use;
    }
  }
  const Leaf& leaf = leaves_[best->leaf];
  owner.definition = leaf.definition;
  owner.occurrences = leaf.occurrences;
  const auto named = named_faces_.find(face.TShape().get());
  owner.face = named == named_faces_.end() ? TDF_Label() : named->second;
  return true;
}

} // namespace occt_backend
} // namespace runmat_geometry_io
