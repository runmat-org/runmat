// Face patterns and indents for K1.
//
// A face pattern repeats the material a set of faces bounds. The faces are removed from a copy
// of the body and the neighbours healed (BRepAlgoAPI_Defeaturing), which gives the body
// without that feature. The difference body \ healed is the material the faces add (a boss);
// healed \ body is the material they remove (a pocket). Each instance moves both pieces
// (BRepBuilderAPI_Transform), fuses the added piece and cuts the removed one. Instance faces
// are copies of the seed faces, tagged by instance, as in a body pattern.
//
// An indent cuts a tool body, grown by a clearance (BRepOffset_MakeOffset; each grown face
// keeps the origins of the tool face it offsets), out of a
// target body; the tool body is kept.

#include "runmat-geometry-io/src/occt/kernel_ffi.rs.h"
#include "exact_kernel.hxx"

#include <BRepAlgoAPI_Defeaturing.hxx>
#include <BRepAlgo_Image.hxx>
#include <BRepOffset_MakeOffset.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopExp_Explorer.hxx>

#include <cstdint>
#include <utility>
#include <vector>

namespace runmat_geometry_io {
namespace occt_kernel {

Tracked transform_instance(const Tracked& src, const gp_Trsf& t, std::int64_t instance);

namespace {
bool has_solid(const TopoDS_Shape& s) {
  return TopExp_Explorer(s, TopAbs_SOLID).More();
}
}  // namespace

Tracked pattern_faces(const Tracked& src, const std::vector<TopoDS_Face>& faces,
                      const std::vector<std::pair<gp_Trsf, std::int64_t>>& instances) {
  if (faces.empty()) fail("a face pattern needs faces");
  BRepAlgoAPI_Defeaturing df;
  df.SetShape(src.shape);
  for (const TopoDS_Face& f : faces) df.AddFaceToRemove(f);
  df.SetRunParallel(Standard_False);
  df.SetToFillHistory(Standard_True);
  df.Build();
  if (!df.IsDone() || df.HasErrors())
    fail("the faces don't bound a feature that can be lifted off the body, so they can't be "
         "repeated; pick every face of the boss or pocket");
  Tracked base;
  base.shape = df.Shape();
  base.sheet = src.sheet;
  propagate_history(src, df.History(), base);
  const Tracked added = boolean_op(1, {&src}, {&base});
  const Tracked removed = boolean_op(1, {&base}, {&src});
  const bool adds = has_solid(added.shape);
  const bool removes = has_solid(removed.shape);
  if (!adds && !removes) fail("the faces bound no material to repeat");
  std::vector<Tracked> add_copies, cut_copies;
  for (const auto& inst : instances) {
    if (adds) add_copies.push_back(transform_instance(added, inst.first, inst.second));
    if (removes) cut_copies.push_back(transform_instance(removed, inst.first, inst.second));
  }
  Tracked out = src;
  if (!add_copies.empty()) {
    std::vector<const Tracked*> tools;
    for (const Tracked& t : add_copies) tools.push_back(&t);
    out = boolean_op(0, {&src}, tools);
  }
  if (!cut_copies.empty()) {
    std::vector<const Tracked*> tools;
    for (const Tracked& t : cut_copies) tools.push_back(&t);
    const Tracked current = out;
    out = boolean_op(1, {&current}, tools);
  }
  return out;
}

Tracked indent_cut(const Tracked& target, const Tracked& tool, double clearance) {
  if (clearance <= 0.0) return boolean_op(1, {&target}, {&tool});
  BRepOffset_MakeOffset mk;
  mk.Initialize(tool.shape, clearance, 1e-7, BRepOffset_Skin, Standard_False, Standard_False,
                GeomAbs_Intersection, Standard_False, Standard_False);
  mk.MakeOffsetShape();
  if (!mk.IsDone() || mk.Shape().IsNull()) fail("the tool body could not be grown by the clearance");
  Tracked grown;
  grown.shape = mk.Shape();
  // Each grown face takes the origins of the tool face it offsets.
  TopTools_IndexedMapOfShape faces;
  TopExp::MapShapes(grown.shape, TopAbs_FACE, faces);
  const BRepAlgo_Image& image = mk.OffsetFacesFromShapes();
  for (int i = 1; i <= tool.faces.Extent(); ++i) {
    const TopoDS_Shape& f = tool.faces.FindKey(i);
    if (!image.HasImage(f)) continue;
    TopTools_ListOfShape imgs;
    image.LastImage(f, imgs);
    for (const TopoDS_Shape& g : imgs)
      for (TopExp_Explorer ex(g, TopAbs_FACE); ex.More(); ex.Next())
        if (faces.Contains(ex.Current())) add_origins(grown.faces, ex.Current(), tool.faces(i));
  }
  return boolean_op(1, {&target}, {&grown});
}

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
