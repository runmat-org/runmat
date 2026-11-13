#pragma once

#include "rust/cxx.h"

#include <TDF_Label.hxx>
#include <TopoDS_Shape.hxx>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

class TopoDS_TShape;

namespace runmat_geometry_io {
namespace occt_backend {

// Names a CAD author gave to solids, faces, edges and vertices (STEP
// `MANIFOLD_SOLID_BREP('Coil', ...)`, product names, `ADVANCED_FACE('Fixed end', ...)`,
// `EDGE_CURVE('Girder', ...)`, `VERTEX_POINT('Left base', ...)`), keyed by kernel shape so they
// survive placement. Names are selector metadata: they never enter persistent identity.
// Faces, edges and vertices are boundary sub-shapes: their names are read, carried and published
// the same way; a kernel shape's own type tells which entity kind it names.
class ExactNameTable {
public:
  // Reads the XCAF sub-shape labels of one definition. Solids without their own name take the
  // definition's product name, or the occurrence's component name when the product is unnamed.
  static ExactNameTable from_definition(const TDF_Label& definition, const TDF_Label& occurrence);

  // `subshape` is a face, edge or vertex.
  void add_subshape_name(const TopoDS_Shape& subshape, const std::string& name);
  void add_solid_name(const TopoDS_Shape& solid, const std::string& name);
  void add_subshape_tag(const TopoDS_Shape& subshape, const std::string& tag);
  void add_solid_tag(const TopoDS_Shape& solid, const std::string& tag);
  void set_body_name(const std::string& name);

  const std::vector<std::string>& subshape_names(const TopoDS_Shape& subshape) const;
  std::vector<std::string> solid_names(const TopoDS_Shape& solid) const;
  // Tags are the presentation layers a CAD tool assigned (STEP PRESENTATION_LAYER_ASSIGNMENT);
  // a solid also carries the layers of its part.
  const std::vector<std::string>& subshape_tags(const TopoDS_Shape& subshape) const;
  std::vector<std::string> solid_tags(const TopoDS_Shape& solid) const;
  const std::string& body_name() const { return body_name_; }

  // The same names and tags carried onto the sub-shapes and solids `image` maps `source`'s onto.
  ExactNameTable transferred(const TopoDS_Shape& source,
                             const std::function<TopoDS_Shape(const TopoDS_Shape&)>& image) const;

private:
  static void add(std::unordered_map<const TopoDS_TShape*, std::vector<std::string>>& map,
                  const TopoDS_Shape& shape,
                  const std::string& name);

  std::unordered_map<const TopoDS_TShape*, std::vector<std::string>> subshapes_;
  std::unordered_map<const TopoDS_TShape*, std::vector<std::string>> solids_;
  std::unordered_map<const TopoDS_TShape*, std::vector<std::string>> subshape_tags_;
  std::unordered_map<const TopoDS_TShape*, std::vector<std::string>> solid_tags_;
  std::vector<std::string> body_tags_;
  // Keeps every named kernel shape alive for as long as its address keys the maps above.
  std::vector<TopoDS_Shape> retained_;
  std::string body_name_;
};

std::string exact_label_name(const TDF_Label& label);

void append_names(rust::Vec<rust::String>& destination, const std::vector<std::string>& names);

} // namespace occt_backend
} // namespace runmat_geometry_io
