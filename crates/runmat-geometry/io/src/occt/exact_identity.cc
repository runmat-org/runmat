#include "runmat-geometry-io/src/occt/ffi.rs.h"
#include "exact_identity.hxx"

#include <BRepTools.hxx>
#include <Message_ProgressIndicator.hxx>
#include <TopAbs_Orientation.hxx>
#include <TopoDS_Shape.hxx>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace runmat_geometry_io {
namespace occt_backend {
namespace {

class CancelProgress final : public Message_ProgressIndicator {
public:
  explicit CancelProgress(const OcctImportOptions& options) : options_(options) {}

protected:
  Standard_Boolean UserBreak() override {
    return options_.cancel_token_id != 0 && occt_import_cancelled(options_.cancel_token_id);
  }

  void Show(const Message_ProgressScope&, const Standard_Boolean) override {}

private:
  OcctImportOptions options_;
};

} // namespace

std::array<std::uint8_t, 32> exact_identity_digest(
    const TopoDS_Shape& shape,
    const OcctImportOptions& options,
    std::uint64_t& serialized_byte_work) {
  std::ostringstream stream;
  CancelProgress progress(options);
  const TopoDS_Shape canonical = shape.Oriented(TopAbs_FORWARD);
  BRepTools::Write(canonical, stream, progress.Start());
  if (options.cancel_token_id != 0 && occt_import_cancelled(options.cancel_token_id)) {
    throw std::runtime_error("OCCT CAD import cancelled");
  }
  const std::string bytes = stream.str();
  if (bytes.empty()) {
    throw std::runtime_error("OCCT exact persistent identity serialization produced no bytes");
  }
  if (serialized_byte_work > options.max_exact_identity_bytes ||
      static_cast<std::uint64_t>(bytes.size()) >
          options.max_exact_identity_bytes - serialized_byte_work) {
    throw std::runtime_error("OCCT exact persistent identity exceeded its byte-work budget");
  }
  serialized_byte_work += static_cast<std::uint64_t>(bytes.size());
  rust::Vec<std::uint8_t> digest = occt_exact_identity_digest(
      rust::Slice<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
  if (digest.size() != 32) {
    throw std::runtime_error("OCCT exact persistent identity digest has an invalid length");
  }
  std::array<std::uint8_t, 32> result;
  std::copy(digest.begin(), digest.end(), result.begin());
  return result;
}

void ExactIdentityContext::append(rust::Vec<std::uint8_t>& destination,
                                  const TopoDS_Shape& shape,
                                  std::uint64_t shape_key,
                                  const OcctImportOptions& options) {
  const auto existing = digests_.find(shape_key);
  if (existing != digests_.end()) {
    for (const std::uint8_t byte : existing->second) {
      destination.push_back(byte);
    }
    return;
  }
  const std::array<std::uint8_t, 32> cached =
      exact_identity_digest(shape, options, serialized_byte_work_);
  for (const std::uint8_t byte : cached) {
    destination.push_back(byte);
  }
  digests_.emplace(shape_key, cached);
}

} // namespace occt_backend
} // namespace runmat_geometry_io
