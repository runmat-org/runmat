#pragma once

// K1: the geometry-engine (kernel) interface over OCCT.
//
// Rust lowers one canonical kernel operation into a short program of instructions
// (`KInstr`); `kernel_execute` interprets it over bodies held in a process registry and
// returns the output bodies with their topology summaries and, per face (and per boundary
// edge of a sheet), the origins it came from. Origins are opaque to C++: an operand face or
// edge, a label id the Rust side assigned from the canonical operation, a transformed copy,
// or a name read from an imported CAD file. History is derived only from OCCT's
// Modified/Generated/IsDeleted records; an output face without an origin fails the operation
// (`history:` errors), it is never guessed.

#include "rust/cxx.h"

#include <BRepBuilderAPI_MakeShape.hxx>
#include <BRepTools_History.hxx>
#include <NCollection_IndexedDataMap.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Vertex.hxx>
#include <gp_Ax3.hxx>
#include <gp_Trsf.hxx>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace runmat_geometry_io {
namespace occt_kernel {

struct KInstr;
struct KOrigin;
struct KResult;
struct KBody;
struct KProjected;

// Error prefixes the Rust side maps to the kernel error taxonomy.
[[noreturn]] void fail(const std::string& message);         // KernelError::Failed
[[noreturn]] void unsupported(const std::string& message);  // KernelError::NotSupported
[[noreturn]] void incomplete(const std::string& message);   // KernelError::HistoryIncomplete

enum OriginKind : std::uint8_t {
  kInput = 0,  // a = operand, b = entity index in the operand's summary
  kLabel = 1,  // a = label id
  kCopy = 2,   // a = operand, b = entity index, c = instance id
  kName = 3,   // a = index into the result's names
};

struct Origin {
  std::uint8_t kind = kLabel;
  std::uint32_t a = 0;
  std::uint32_t b = 0;
  std::uint32_t c = 0;
  bool operator<(const Origin& o) const {
    if (kind != o.kind) return kind < o.kind;
    if (a != o.a) return a < o.a;
    if (b != o.b) return b < o.b;
    return c < o.c;
  }
  bool operator==(const Origin& o) const {
    return kind == o.kind && a == o.a && b == o.b && c == o.c;
  }
};

using Origins = std::vector<Origin>;  // sorted, unique
void merge_into(Origins& into, const Origins& from);
Origins label(std::uint32_t id);

using ShapeOrigins = NCollection_IndexedDataMap<TopoDS_Shape, Origins, TopTools_ShapeMapHasher>;

// A shape under construction with the origins of its faces, edges and vertices.
struct Tracked {
  TopoDS_Shape shape;
  ShapeOrigins faces;
  ShapeOrigins edges;
  ShapeOrigins vertices;
  // Faces that are the operand face itself (same TShape), for `Evolution::Kept`.
  NCollection_IndexedDataMap<TopoDS_Shape, int, TopTools_ShapeMapHasher> kept;
  bool sheet = false;
};

// A body the kernel made: the shape and the summary lists its entity indices refer to.
struct Body {
  TopoDS_Shape shape;
  std::vector<TopoDS_Face> faces;
  std::vector<TopoDS_Edge> edges;      // summary edges (no seams, no degenerate edges)
  std::vector<TopoDS_Vertex> vertices;  // summary vertices
  bool sheet = false;
};

std::shared_ptr<const Body> registry_get(std::uint64_t id);
std::uint64_t registry_put(std::shared_ptr<Body> body);
std::shared_ptr<Body> summarize(const TopoDS_Shape& shape, bool sheet);

// Origins of a tracked entity; empty when none.
const Origins& origins_of(const ShapeOrigins& map, const TopoDS_Shape& s);
void add_origins(ShapeOrigins& map, const TopoDS_Shape& s, const Origins& o);

// Carry face, edge and vertex origins from `in` to `result` through a history.
// Modified images inherit; unmodified, undeleted entities still present keep theirs.
void propagate(const Tracked& in, BRepBuilderAPI_MakeShape& h, Tracked& out);
void propagate_history(const Tracked& in, const Handle(BRepTools_History) & h, Tracked& out);

// A tracked operand: every face, edge and vertex carries its Input origin.
Tracked track_operand(const Body& body, std::uint32_t operand);

// Same-domain merge per the contract's flush-face rule, then deterministic output bodies.
struct MergePolicy {
  bool merge_collinear = false;
  std::vector<std::uint32_t> side_labels;  // labels of profile side faces (sorted)
};
Tracked merge_same_domain(const Tracked& in, const MergePolicy& policy);

// Frames and transforms from the flat real encoding.
gp_Ax3 frame_from(const double* r);  // origin, x, y, z (12 reals)
gp_Trsf trsf_from(const double* r);  // rows of R then t (12 reals)

// Builders shared across files.
struct ProfileCap {
  TopoDS_Edge edge;
  std::uint32_t start_cap;  // boundary label where a sheet's start cap would be
  std::uint32_t end_cap;    // boundary label at the end cap
};
struct Profile {
  Tracked tracked;  // the profile face (closed loops) or a compound of wires (open chains)
  gp_Ax3 frame;
  bool closed = true;
  std::vector<ProfileCap> caps;  // per segment, in order
};
// Profile encoding: see exact_sketch_profile.cc.
Profile build_profile(const KInstr& in);

Tracked build_prism(const Profile& p, double start, double end, std::uint32_t start_label,
                    std::uint32_t end_label, double draft_end = 0.0, double draft_start = 0.0);
Tracked build_revolve(const Profile& p, const gp_Ax1& axis, double angle,
                      std::uint32_t start_label, std::uint32_t end_label);
Tracked build_sweep(const Profile& p, const std::vector<gp_Pnt>& path, std::uint32_t start_label,
                    std::uint32_t end_label);
Tracked build_loft(const std::vector<const Profile*>& sections, std::uint32_t start_label,
                   std::uint32_t end_label);
Tracked boolean_op(int op, const std::vector<const Tracked*>& args,
                   const std::vector<const Tracked*>& tools);

constexpr std::uint32_t kNoLabel = 0xffffffffu;

// Exact queries used by the Rust lowering before it builds a program.
rust::Vec<double> kernel_ray_hits(rust::Slice<const std::uint64_t> bodies,
                                  rust::Slice<const std::int64_t> face_filter,
                                  rust::Slice<const double> ray);
rust::Vec<double> kernel_extent(rust::Slice<const std::uint64_t> bodies,
                                rust::Slice<const double> axis);

KResult kernel_execute(rust::Slice<const KInstr> program,
                       rust::Slice<const std::uint64_t> operands);
void kernel_release(std::uint64_t id);
std::uint64_t kernel_live_bodies();
rust::Vec<double> kernel_analyze(std::uint32_t kind, rust::Slice<const double> reals,
                                 rust::Slice<const std::uint64_t> bodies);
rust::Vec<KProjected> kernel_project(rust::Slice<const std::uint64_t> bodies,
                                     rust::Slice<const double> transforms,
                                     rust::Slice<const double> frame, std::uint32_t flags,
                                     double tolerance);
rust::Vec<double> kernel_proximity(std::uint64_t a, rust::Slice<const double> ta,
                                   std::uint64_t b, rust::Slice<const double> tb);
rust::Vec<std::uint8_t> kernel_export_step(rust::Slice<const KInstr> document,
                                           rust::Slice<const std::uint64_t> bodies);
rust::String kernel_occt_version();
rust::Vec<rust::String> kernel_step_products(rust::Str path);

}  // namespace occt_kernel
}  // namespace runmat_geometry_io
