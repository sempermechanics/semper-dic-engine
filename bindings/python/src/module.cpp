// pybind11 bindings for the Semper DIC engine — wraps the C++ core directly.
// NumPy in, (N, 8) points + (24,) metrics out. Behavior is identical to the
// Android/JNI path; see docs/CONTRACT.md for the frozen formats.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <semper/incremental.hpp>
#include <semper/io.hpp>
#include <semper/pipeline.hpp>
#include <semper/version.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using Semper::pipeline::CancelScope;
using Semper::pipeline::CancelToken;
using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ProgressCallback;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;

namespace {

// The object whose progress callback this thread is running, if any.
thread_local const void* t_in_callback_of = nullptr;

// The first Python exception raised while a solve ran -- by its progress
// callback, or by a signal handler (Ctrl-C) -- kept to be raised once the
// solve has stopped. Touched only with the GIL held.
struct SolveErrors {
    std::unique_ptr<py::error_already_set> err;
    void keep(py::error_already_set&& e) {
        if (!err) err = std::make_unique<py::error_already_set>(std::move(e));
    }
};

ProgressCallback to_callback(const py::object& progress, const void* owner, SolveErrors& errors,
                             CancelToken& cancel);

// Runs `solve` on a worker thread and waits for it here with the GIL
// released, waking every 50 ms to run Python's signal handlers. A Ctrl-C
// (KeyboardInterrupt), or any handler that raises, cancels the solve; the
// exception -- or one the progress callback raised -- is raised here once the
// solve has stopped. The solve used to run on this thread with the GIL
// released, so nothing delivered a Ctrl-C until it had finished.
template <class Solve>
int run_interruptible(CancelToken& cancel, SolveErrors& errors, Solve&& solve) {
    int result = 0;
    std::exception_ptr failure;
    {
        py::gil_scoped_release rel;
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
        std::thread worker([&] {
            try {
                result = solve();
            } catch (...) {
                failure = std::current_exception();
            }
            std::lock_guard<std::mutex> lk(m);
            done = true;
            cv.notify_one();
        });
        std::unique_lock<std::mutex> lk(m);
        while (!cv.wait_for(lk, std::chrono::milliseconds(50), [&] { return done; })) {
            lk.unlock();
            {
                py::gil_scoped_acquire gil;
                if (PyErr_CheckSignals() != 0) {
                    errors.keep(py::error_already_set());
                    cancel.request();
                }
            }
            lk.lock();
        }
        lk.unlock();
        worker.join();
    }
    if (errors.err) {
        py::error_already_set e = std::move(*errors.err);
        errors.err.reset();
        throw e;
    }
    if (failure) std::rethrow_exception(failure);
    return result;
}

// The output buffer for a grid of `points` nodes, 8 floats each, refusing one
// the solver could not address: its capacity is an int, and a larger buffer
// used to be truncated to a negative capacity that wrote nothing.
size_t output_floats(size_t points) {
    if (points > (size_t)INT_MAX / 8) {
        throw std::length_error("grid too large: more than INT_MAX / 8 points for one output buffer");
    }
    return std::max<size_t>(points, 1) * 8;
}

// Accepts a 2-D ndarray (H, W) — copied so it owns its pixels across the
// solve — or encoded bytes (PNG/JPEG) decoded via io. A uint8 array is taken
// as is; any other dtype (uint16, float in [0, 1], ...) is mapped from its own
// range onto 0..255 by io::to_gray8, as a decoded 16-bit file is. (Until
// 0.5.0 pybind11's forcecast truncated it to uint8: a [0, 1] float image
// became black.)
cv::Mat to_gray(const py::object& img, int expected_w, int expected_h) {
    if (py::isinstance<py::bytes>(img) || py::isinstance<py::bytearray>(img)) {
        std::string s = py::cast<std::string>(img);
        return Semper::io::decode_gray(reinterpret_cast<const uint8_t*>(s.data()),
                                       s.size(), expected_w, expected_h);
    }
    auto check_2d = [](const py::array& a) {
        if (a.ndim() != 2) throw std::invalid_argument("grayscale image must be 2-D (H, W)");
        if (a.shape(0) > INT32_MAX || a.shape(1) > INT32_MAX)
            throw std::invalid_argument("image too large");
    };
    if (py::isinstance<py::array_t<uint8_t>>(img)) {
        auto arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(img);
        check_2d(arr);
        cv::Mat wrapped(static_cast<int>(arr.shape(0)), static_cast<int>(arr.shape(1)), CV_8UC1,
                        const_cast<uint8_t*>(arr.data()));
        return wrapped.clone();
    }
    auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(img);
    if (!arr) throw std::invalid_argument("image must be a numeric ndarray or encoded bytes");
    check_2d(arr);
    cv::Mat wrapped(static_cast<int>(arr.shape(0)), static_cast<int>(arr.shape(1)), CV_64FC1,
                    const_cast<double*>(arr.data()));
    return Semper::io::to_gray8(wrapped);
}

// None is no mask; a mask that does not decode raises. Solving without it
// would report a field over the regions the caller excluded.
//
// The engine excludes a pixel whose 8-bit mask value is below 128. A mask is
// not an image, so it is never stretched like one (io::to_gray8 turned an
// all-False bool mask into 128, i.e. everything included):
//   - bool, or any non-uint8 dtype: nonzero is included, zero excluded;
//   - uint8 holding only 0 and 1 -- (labels > 0).astype(np.uint8) -- is the
//     same boolean mask (it used to exclude every pixel and return -2);
//   - any other uint8 mask, and encoded bytes, keep the < 128 rule
//     (io::decode_mask makes 16-bit and float files boolean).
cv::Mat to_mask(const py::object& mask, int expected_w, int expected_h) {
    if (mask.is_none()) return cv::Mat();
    cv::Mat roi;
    if (py::isinstance<py::bytes>(mask) || py::isinstance<py::bytearray>(mask)) {
        std::string s = py::cast<std::string>(mask);
        roi = Semper::io::decode_mask(reinterpret_cast<const uint8_t*>(s.data()), s.size(),
                                      expected_w, expected_h);
    } else {
        auto check_2d = [](const py::array& a) {
            if (a.ndim() != 2) throw std::invalid_argument("mask must be 2-D (H, W)");
            if (a.shape(0) > INT32_MAX || a.shape(1) > INT32_MAX)
                throw std::invalid_argument("mask too large");
        };
        if (py::isinstance<py::array_t<uint8_t>>(mask)) {
            auto arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(mask);
            check_2d(arr);
            cv::Mat wrapped(static_cast<int>(arr.shape(0)), static_cast<int>(arr.shape(1)), CV_8UC1,
                            const_cast<uint8_t*>(arr.data()));
            double hi = 0.0;
            cv::minMaxLoc(wrapped, nullptr, &hi);
            roi = hi <= 1.0 ? Semper::io::to_mask8(wrapped == 1) : wrapped.clone();
        } else {
            // bool and every other numeric dtype: nonzero is included.
            auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(mask);
            if (!arr) throw std::invalid_argument("mask must be a numeric ndarray or encoded bytes");
            check_2d(arr);
            cv::Mat wrapped(static_cast<int>(arr.shape(0)), static_cast<int>(arr.shape(1)), CV_64FC1,
                            const_cast<double*>(arr.data()));
            roi = Semper::io::to_mask8(wrapped);
        }
    }
    if (roi.empty()) throw std::invalid_argument("failed to decode mask");
    return roi;
}

// Serialises the calls on one object, cancel() aside: a solve releases the
// GIL, and another thread's set_reference() would free the reference under
// it. Taken with the GIL released, because the solve holding it re-enters
// Python for progress -- waiting for it under the GIL would deadlock. A call
// from the object's own progress callback would wait for the solve that
// called it, forever: it raises instead.
std::unique_lock<std::mutex> lock_without_gil(std::mutex& m, const void* owner) {
    if (t_in_callback_of == owner) {
        throw std::runtime_error("cannot call this object's methods from inside its own progress "
                                 "callback (only cancel() is allowed there)");
    }
    py::gil_scoped_release rel;
    return std::unique_lock<std::mutex>(m);
}

class Engine {
public:
    Engine() = default;

    void set_reference(const py::object& img, const py::object& mask,
                       int expected_w, int expected_h) {
        auto lock = lock_without_gil(mutex_, this);
        cv::Mat ref = to_gray(img, expected_w, expected_h);
        if (ref.empty()) throw std::runtime_error("failed to decode reference image");
        cv::Mat roi = to_mask(mask, ref.cols, ref.rows);
        // Gradients and mask for the whole image: no Python object is
        // touched, so other Python threads run meanwhile.
        py::gil_scoped_release rel;
        cache_.set_from_gray(ref, roi);
    }

    // Returns (code, points|None, metrics). code >= 0 is the point count; negative
    // is a SEMPER error (-2 ROI, -3 init, -4 strain window, -5 seed, -99 cancelled). points is (code, 8) f32.
    py::tuple run(const py::object& deformed, const py::object& mask,
                  std::array<int, 4> rect, int step, int subset, int strain_window,
                  bool use_6x6, const py::object& progress) {
        auto lock = lock_without_gil(mutex_, this);
        // From here on a cancel() stops this solve, the decode and the output
        // allocation below included (they used to run before the solve
        // cleared the token, which dropped a cancel issued meanwhile).
        CancelScope cancel_window(cancel_);
        cv::Mat def = to_gray(deformed, cache_.width, cache_.height);
        if (def.empty()) throw std::runtime_error("failed to decode deformed image");
        cv::Mat roi = to_mask(mask, cache_.width, cache_.height);

        FullFieldParams p;
        p.rect_x = rect[0];
        p.rect_y = rect[1];
        p.rect_w = rect[2];
        p.rect_h = rect[3];
        p.step = step;
        p.subset_size = subset;
        p.strain_window = strain_window;
        p.use_6x6_interpolator = use_6x6;

        // One point per grid node is the upper bound; 8 floats each. Over-allocate
        // so the solver never has to drop points (see the capacity contract).
        // The engine clips the rect to the image, so the image bounds the grid
        // too: an oversized rect must not size this buffer.
        // Parameters the solve will refuse get a one-point buffer: it returns
        // their error code without writing, and an invalid call no longer
        // allocates for the whole grid first (gigabytes at step 1).
        const bool plausible = step > 0 && subset >= 3 && subset % 2 == 1 && p.rect_w > 0 &&
                               p.rect_h > 0 && cache_.width > 0;
        const int span_w = std::min(p.rect_w, cache_.width);
        const int span_h = std::min(p.rect_h, cache_.height);
        const size_t cols = plausible ? static_cast<size_t>(std::max(0, span_w / step + 1)) : 0;
        const size_t rows = plausible ? static_cast<size_t>(std::max(0, span_h / step + 1)) : 0;
        // At least one point: a zero-sized buffer has no data() and the solve
        // would report it as -3 before validating the step or rect (-2).
        std::vector<float> out(output_floats(cols * rows), 0.0f);
        float metrics[Semper::pipeline::kMetricsLen] = {0};

        SolveErrors errors;
        ProgressCallback cb = to_callback(progress, this, errors, cancel_);

        const int n = run_interruptible(cancel_, errors, [&] {
            return run_full_field(cache_, def, roi, p, out.data(), static_cast<int>(out.size()),
                                  metrics, Semper::pipeline::kMetricsLen, cancel_, cb);
        });

        py::array_t<float> met(Semper::pipeline::kMetricsLen);
        std::memcpy(met.mutable_data(), metrics, sizeof(metrics));
        if (n < 0) {
            return py::make_tuple(n, py::none(), met);
        }
        py::array_t<float> points({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(8)});
        std::memcpy(points.mutable_data(), out.data(),
                    static_cast<size_t>(n) * 8 * sizeof(float));
        return py::make_tuple(n, points, met);
    }

    void cancel() { cancel_.request(); }

private:
    std::mutex mutex_;
    ReferenceCache cache_;
    CancelToken cancel_;
};

ProgressCallback to_callback(const py::object& progress, const void* owner, SolveErrors& errors,
                             CancelToken& cancel) {
    ProgressCallback cb;
    if (!progress.is_none()) {
        // Checked here: an uncallable object used to raise TypeError on every
        // tick, on the solver's thread, where it was logged and ignored.
        if (!PyCallable_Check(progress.ptr())) throw py::type_error("progress must be callable or None");
        // The solve copies and destroys the std::function with the GIL
        // released, so it must not hold a py::function by value: its
        // refcount would change without the GIL. A shared_ptr counts
        // atomically, and the last owner takes the GIL to release it.
        std::shared_ptr<py::function> fn(
                new py::function(py::reinterpret_borrow<py::function>(progress)),
                [](py::function* f) {
                    py::gil_scoped_acquire gil;
                    delete f;
                });
        SolveErrors* errs = &errors;   // outlives the solve: run_interruptible joins it
        CancelToken* token = &cancel;
        cb = [fn, owner, errs, token](int pct) {
            py::gil_scoped_acquire gil;  // re-enter Python to call back
            t_in_callback_of = owner;
            try {
                (*fn)(pct);
            } catch (py::error_already_set& e) {
                // An exception in the callback stops the solve and is raised
                // to the caller (it used to be logged and the solve went on).
                errs->keep(std::move(e));
                token->request();
            }
            t_in_callback_of = nullptr;
        };
    }
    return cb;
}

// Incremental (updated-reference) DIC over a frame sequence; see
// <semper/incremental.hpp>. Results are total displacement from the first
// image, at its grid points, in the same (N, 8) layout as Engine.run.
class Incremental {
public:
    Incremental() = default;

    int init(const py::object& img, const py::object& mask, std::array<int, 4> rect,
             int step, int subset, int strain_window, bool use_6x6, int update_interval,
             int expected_w, int expected_h) {
        auto lock = lock_without_gil(mutex_, this);
        cv::Mat ref = to_gray(img, expected_w, expected_h);
        if (ref.empty()) throw std::runtime_error("failed to decode reference image");
        cv::Mat roi = to_mask(mask, ref.cols, ref.rows);
        Semper::pipeline::IncrementalParams ip;
        ip.field.rect_x = rect[0];
        ip.field.rect_y = rect[1];
        ip.field.rect_w = rect[2];
        ip.field.rect_h = rect[3];
        ip.field.step = step;
        ip.field.subset_size = subset;
        ip.field.strain_window = strain_window;
        ip.field.use_6x6_interpolator = use_6x6;
        ip.update_interval = update_interval;
        int rc;
        {
            py::gil_scoped_release rel;   // the reference's gradients and mask scan
            rc = session_.init(ref, roi, ip);
        }
        width_ = rc == 0 ? ref.cols : 0;
        height_ = rc == 0 ? ref.rows : 0;
        return rc;
    }

    // Returns (code, points|None, metrics, info dict).
    py::tuple process(const py::object& deformed, const py::object& progress) {
        auto lock = lock_without_gil(mutex_, this);
        CancelScope cancel_window(cancel_);   // as Engine.run
        cv::Mat def = to_gray(deformed, width_, height_);
        if (def.empty()) throw std::runtime_error("failed to decode deformed image");
        std::vector<float> out(output_floats(static_cast<size_t>(std::max(session_.max_points(), 0))), 0.0f);
        float metrics[Semper::pipeline::kMetricsLen] = {0};
        Semper::pipeline::IncrementalFrameInfo fi;
        SolveErrors errors;
        ProgressCallback cb = to_callback(progress, this, errors, cancel_);

        const int n = run_interruptible(cancel_, errors, [&] {
            return session_.process_frame(def, out.data(), static_cast<int>(out.size()), metrics,
                                          Semper::pipeline::kMetricsLen, cancel_, cb, &fi);
        });

        py::array_t<float> met(Semper::pipeline::kMetricsLen);
        std::memcpy(met.mutable_data(), metrics, sizeof(metrics));
        // An error solved no frame and leaves the session as it was: no info
        // (it used to be a default one, all zeros, describing nothing).
        if (n < 0) return py::make_tuple(n, py::none(), met, py::none());
        py::dict info;
        info["reference_frame"] = fi.reference_frame;
        info["reference_updated"] = fi.reference_updated;
        info["tracked_points"] = fi.tracked_points;
        info["frame0_points"] = fi.frame0_points;
        info["taylor_fallbacks"] = fi.taylor_fallbacks;
        info["lost_this_frame"] = fi.lost_this_frame;
        py::array_t<float> points({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(8)});
        std::memcpy(points.mutable_data(), out.data(), static_cast<size_t>(n) * 8 * sizeof(float));
        return py::make_tuple(n, points, met, info);
    }

    void cancel() { cancel_.request(); }
    int frames_processed() {
        auto lock = lock_without_gil(mutex_, this);
        return session_.frames_processed();
    }

private:
    std::mutex mutex_;
    Semper::pipeline::IncrementalSession session_;
    CancelToken cancel_;
    int width_ = 0, height_ = 0;
};

}  // namespace

PYBIND11_MODULE(_semper, m) {
    m.doc() = "Semper DIC engine (native bindings)";
    m.attr("__version__") = SEMPER_VERSION_STRING;

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def("set_reference", &Engine::set_reference,
             py::arg("image"), py::arg("mask") = py::none(),
             py::arg("expected_w") = 0, py::arg("expected_h") = 0)
        .def("run", &Engine::run,
             py::arg("deformed"), py::arg("mask") = py::none(),
             py::arg("rect"), py::arg("step"), py::arg("subset"),
             py::arg("strain_window"), py::arg("use_6x6") = false,
             py::arg("progress") = py::none())
        .def("cancel", &Engine::cancel);

    py::class_<Incremental>(m, "IncrementalSession")
        .def(py::init<>())
        .def("init", &Incremental::init,
             py::arg("image"), py::arg("mask") = py::none(),
             py::arg("rect"), py::arg("step"), py::arg("subset"),
             py::arg("strain_window"), py::arg("use_6x6") = false,
             py::arg("update_interval") = 1,
             py::arg("expected_w") = 0, py::arg("expected_h") = 0)
        .def("process", &Incremental::process,
             py::arg("deformed"), py::arg("progress") = py::none())
        .def("cancel", &Incremental::cancel)
        .def_property_readonly("frames_processed", &Incremental::frames_processed);
}
