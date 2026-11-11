#pragma once

#include <Standard_Handle.hxx>
#include <TDF_Label.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Trsf.hxx>

#include <cstddef>
#include <unordered_map>
#include <vector>

class TopoDS_TShape;
class XCAFDoc_ShapeTool;

namespace runmat_geometry_io {
namespace occt_backend {

// The XCAF labels that own a preview face.
struct PreviewFaceOwner {
  // The face's own named sub-shape label (`ADVANCED_FACE('root', ...)`), or null when the author
  // left the face unnamed.
  TDF_Label face;
  // The part definition (product) whose shape holds the face.
  TDF_Label definition;
  // The occurrence labels from the free shape down to the part's component, in the same order as
  // the exact import's occurrence path.
  std::vector<TDF_Label> occurrences;
};

// Finds the owners of a face taken from the document's located compound. A face inside an
// assembly product carries its occurrence's placement, so XCAF's located shape search misses it.
// This index matches faces by kernel shape, as the exact import's name table does, and tells
// repeated occurrences of one product apart by their placement.
class PreviewFaceOwners {
public:
  static PreviewFaceOwners build(const Handle(XCAFDoc_ShapeTool)& shape_tool);

  bool resolve(const TopoDS_Face& face, PreviewFaceOwner& owner) const;

private:
  struct Leaf {
    TDF_Label definition;
    std::vector<TDF_Label> occurrences;
    gp_Trsf placement;
  };
  struct Use {
    std::size_t leaf;
    gp_Trsf placement;
  };

  void add_occurrence(const TDF_Label& label,
                      std::vector<TDF_Label> occurrences,
                      const gp_Trsf& parent,
                      std::size_t depth);

  std::vector<Leaf> leaves_;
  std::unordered_map<const TopoDS_TShape*, std::vector<Use>> uses_;
  std::unordered_map<const TopoDS_TShape*, TDF_Label> named_faces_;
  std::vector<TopoDS_Shape> retained_;
};

} // namespace occt_backend
} // namespace runmat_geometry_io
