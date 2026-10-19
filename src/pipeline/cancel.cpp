#include <semper/cancel.hpp>

namespace Semper {
namespace pipeline {

// Its own translation unit, free of OpenCV and threads, so the host test suite
// can link the cancel contract without pulling in the whole solver.
//
// The free functions act on one process-wide token, which the token-less
// overloads of run_full_field and IncrementalSession::process_frame hand to
// the solver. A solve given its own token never looks at this one.

namespace detail {

CancelToken& global_cancel_token() noexcept {
    static CancelToken token;
    return token;
}

} // namespace detail

void request_cancel() { detail::global_cancel_token().request(); }

void clear_cancel() { detail::global_cancel_token().clear(); }

bool cancel_requested() { return detail::global_cancel_token().requested(); }

} // namespace pipeline
} // namespace Semper
