#pragma once

#include <BRepTools.hxx>
#include <BRepTools_ShapeSet.hxx>
#include <TopoDS.hxx>
#include <rust/cxx.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace runmat_geometry_io::occt_backend::detail {

// One fixed inventory per immutable session. Rust admits both the input keys
// and this requested storage before starting native reconstruction.
class ExactSurfaceBounds {
  struct Entry {
    std::uint64_t face_key;
    std::array<double, 4> bounds;
  };
  static_assert(sizeof(Entry) == 40, "update the Rust session storage admission");
  std::vector<Entry> entries_;

public:
  ExactSurfaceBounds() = default;

  ExactSurfaceBounds(const BRepTools_ShapeSet& shapes,
                     rust::Slice<const std::uint64_t> face_keys)
      : entries_(face_keys.size()) {
    for (std::size_t index = 0; index < face_keys.size(); ++index) {
      const auto key = face_keys[index];
      if (key == 0 || key > static_cast<std::uint64_t>(shapes.NbShapes()) ||
          (index > 0 && face_keys[index - 1] >= key)) {
        throw std::runtime_error("OCCT surface cache requires ordered unique face keys");
      }
      const auto& shape = shapes.Shape(static_cast<Standard_Integer>(key));
      if (shape.ShapeType() != TopAbs_FACE) {
        throw std::runtime_error("OCCT surface cache key does not identify a face");
      }
      auto& entry = entries_[index];
      entry.face_key = key;
      BRepTools::UVBounds(TopoDS::Face(shape), entry.bounds[0], entry.bounds[1],
                         entry.bounds[2], entry.bounds[3]);
    }
  }

  std::array<double, 4> at(std::uint64_t key) const {
    const auto found = std::lower_bound(entries_.begin(), entries_.end(), key,
        [](const Entry& entry, std::uint64_t candidate) { return entry.face_key < candidate; });
    if (found == entries_.end() || found->face_key != key) {
      throw std::runtime_error("OCCT exact face is outside the admitted surface inventory");
    }
    return found->bounds;
  }
};

} // namespace runmat_geometry_io::occt_backend::detail
