// pybind11 bindings for the Semper DIC engine — wraps the C++ core directly.
// NumPy in, (N, 8) points + (23,) metrics out. Behavior is identical to the
// Android/JNI path; see docs/CONTRACT.md for the frozen formats.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <semper/io.hpp>
#include <semper/pipeline.hpp>
#include <semper/version.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using Semper::pipeline::CancelToken;
using Semper::pipeline::FullFieldParams;
using Semper::pipeline::ProgressCallback;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;

namespace {

// Accepts a 2-D uint8 ndarray (H, W) — wrapped as grayscale, copied so it owns
// its pixels across the solve — or encoded bytes (PNG/JPEG) decoded via io.
cv::Mat to_gray(const py::object& img, int expected_w, int expected_h) {
    if (py::isinstance<py::bytes>(img) || py::isinstance<py::bytearray>(img)) {
        std::string s = py::cast<std::string>(img);
        return Semper::io::decode_gray(reinterpret_cast<const uint8_t*>(s.data()),
                                       s.size(), expected_w, expected_h);
    }
    auto arr = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(img);
    if (!arr) throw std::invalid_argument("image must be a uint8 ndarray or encoded bytes");
    if (arr.ndim() != 2) throw std::invalid_argument("grayscale image must be 2-D (H, W)");
    const int h = static_cast<int>(arr.shape(0));
    const int w = static_cast<int>(arr.shape(1));
    cv::Mat wrapped(h, w, CV_8UC1, const_cast<uint8_t*>(arr.data()));
    return wrapped.clone();
}

class Engine {
public:
    Engine() = default;

    void set_reference(const py::object& img, const py::object& mask,
                       int expected_w, int expected_h) {
        cv::Mat ref = to_gray(img, expected_w, expected_h);
        if (ref.empty()) throw std::runtime_error("failed to decode reference image");
        cv::Mat roi;
        if (!mask.is_none()) roi = to_gray(mask, ref.cols, ref.rows);
        cache_.set_from_gray(ref, roi);
    }

    // Returns (code, points|None, metrics). code >= 0 is the point count; negative
    // is a SEMPER error (-2 ROI, -3 init, -99 cancelled). points is (code, 8) f32.
    py::tuple run(const py::object& deformed, const py::object& mask,
                  std::array<int, 4> rect, int step, int subset, int strain_window,
                  bool use_6x6, const py::object& progress) {
        cv::Mat def = to_gray(deformed, cache_.width, cache_.height);
        if (def.empty()) throw std::runtime_error("failed to decode deformed image");
        cv::Mat roi;
        if (!mask.is_none()) roi = to_gray(mask, cache_.width, cache_.height);

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
        const int cols = step > 0 ? (p.rect_w / step + 1) : 0;
        const int rows = step > 0 ? (p.rect_h / step + 1) : 0;
        const size_t max_pts = static_cast<size_t>(std::max(0, cols) * std::max(0, rows));
        std::vector<float> out(max_pts * 8, 0.0f);
        float metrics[23] = {0};

        ProgressCallback cb;
        if (!progress.is_none()) {
            py::function fn = py::reinterpret_borrow<py::function>(progress);
            cb = [fn](int pct) {
                py::gil_scoped_acquire gil;  // re-enter Python to call back
                fn(pct);
            };
        }

        int n;
        {
            py::gil_scoped_release rel;  // long solve must not hold the GIL
            n = run_full_field(cache_, def, roi, p, out.data(),
                               static_cast<int>(out.size()), metrics, 23, cancel_, cb);
        }

        py::array_t<float> met(23);
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
    ReferenceCache cache_;
    CancelToken cancel_;
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
}
