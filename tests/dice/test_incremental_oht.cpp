// IncrementalSession on real images: DICe's open-hole tension ladder.
//
// DICe ships solved fields for oht_cfrp_01/03/06/11, each against the fixed
// image 00. Feeding 00 -> 01 -> 03 -> 06 -> 11 through an IncrementalSession
// (reference moved every frame) must still give the TOTAL field from image 00
// at DICe's coordinates, so each rung compares against DICe's own fixed-
// reference solution. The increments are 0.9, 2.3, 3.4 and 5.4 px.
//
// The grid is step 7 with origin and extent chosen so every DICe point (step
// 35, from x=61, y=61) is a grid node two steps inside the ROI, which keeps it
// clear of the VSG edge ring. No mask: nodes in the hole fail or are rejected
// as outliers, as in the fixed-reference app path.
//
// Like test_field_agreement.cpp, the metric is AGREEMENT with another code on
// a real experiment. The fixed-reference full-field result on the same grid is
// logged next to it for comparison.
#include "framework/test_framework.h"

#if defined(DIC_HAVE_OPENCV)

#include "framework/image_io.h"

#include <semper/incremental.hpp>
#include <semper/pipeline.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using Semper::pipeline::FullFieldParams;
using Semper::pipeline::IncrementalFrameInfo;
using Semper::pipeline::IncrementalParams;
using Semper::pipeline::IncrementalSession;
using Semper::pipeline::ReferenceCache;
using Semper::pipeline::run_full_field;

namespace {

    // Agreement bounds, set from the measured result. Rung 01 is the first
    // frame and matches the fixed reference exactly (rms 0.0006 px). After that
    // the incremental error grows to rms du 0.058 / dv 0.026 px at rung 11,
    // almost all of it a uniform u bias (mean du -0.054 px) that builds up
    // rung by rung. The u increments are small sub-pixel steps of about
    // -0.2 px each, the regime where each solve's interpolation bias has one
    // sign and adds up. The fixed reference meets DICe to 0.003 px on the same
    // grid. That is the expected trade: on moderate, smooth loading the fixed
    // reference is more accurate. The bound keeps ~1.7x headroom.
    constexpr double INC_RMS_TOL = 0.1;
    constexpr double INC_MIN_FRACTION = 0.90;

    struct Gold {
        int x, y;
        double u, v;
    };

    bool load_gold(const std::string &path, std::vector<Gold> &out) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line.rfind("***", 0) == 0) continue;
            if (line.rfind("COORDINATE_X", 0) == 0) continue;
            std::stringstream ss(line);
            std::string cell;
            double col[4];
            int n = 0;
            while (n < 4 && std::getline(ss, cell, ',')) col[n++] = std::atof(cell.c_str());
            if (n < 4) continue;
            out.push_back({(int) std::lround(col[0]), (int) std::lround(col[1]), col[2], col[3]});
        }
        return !out.empty();
    }

    cv::Mat load_mat(const char *name) {
        dictest::GrayImage g;
        if (!dictest::load_gray(std::string(DICE_FIXTURES_DIR) + name, g)) return cv::Mat();
        return cv::Mat(g.h, g.w, CV_8UC1, g.px.data()).clone();
    }

    FullFieldParams oht_params() {
        FullFieldParams p;
        p.step = 7;
        p.rect_x = 61 - 2 * p.step;
        p.rect_y = 61 - 2 * p.step;
        p.rect_w = (341 - 61) + 4 * p.step;
        p.rect_h = (971 - 61) + 4 * p.step;
        p.subset_size = 27;           // DICe gold header
        p.strain_window = 21;
        p.use_6x6_interpolator = true; // DICe used KEYS_FOURTH
        return p;
    }

    int capacity_for(const FullFieldParams &p) {
        return (p.rect_w / p.step + 1) * (p.rect_h / p.step + 1) * 8;
    }

    struct Agreement {
        int compared = 0;
        size_t gold = 0;
        double mean_u = 0, mean_v = 0, rms_u = 0, rms_v = 0, max_d = 0;
        double frac() const { return gold ? (double) compared / (double) gold : 0.0; }
    };

    Agreement compare(const std::vector<float> &out, int n, const std::vector<Gold> &gold) {
        std::map<std::pair<int, int>, std::pair<float, float>> at;
        for (int i = 0; i < n; ++i) {
            const float *r = &out[(size_t) i * 8];
            at[{(int) std::lround(r[0]), (int) std::lround(r[1])}] = {r[2], r[3]};
        }
        Agreement a;
        a.gold = gold.size();
        double su = 0, sv = 0, su2 = 0, sv2 = 0;
        for (const Gold &g : gold) {
            auto it = at.find({g.x, g.y});
            if (it == at.end()) continue;
            const double du = it->second.first - g.u, dv = it->second.second - g.v;
            su += du;
            sv += dv;
            su2 += du * du;
            sv2 += dv * dv;
            a.max_d = std::max(a.max_d, std::max(std::fabs(du), std::fabs(dv)));
            ++a.compared;
        }
        if (a.compared > 0) {
            a.mean_u = su / a.compared;
            a.mean_v = sv / a.compared;
            a.rms_u = std::sqrt(su2 / a.compared);
            a.rms_v = std::sqrt(sv2 / a.compared);
        }
        return a;
    }

    struct Rung {
        const char *image;
        const char *gold;
    };

    const Rung kLadder[] = {
        {"/oht_cfrp_01.tiff", "/DICe_solution_01.txt"},
        {"/oht_cfrp_03.tiff", "/DICe_solution_03.txt"},
        {"/oht_cfrp_06.tiff", "/DICe_solution_06.txt"},
        {"/oht_cfrp_11.tiff", "/DICe_solution_11.txt"},
    };

} // namespace

TEST_CASE(DiceIncremental, OhtCfrp_LadderTotalsAgreeWithDice) {
    const cv::Mat ref = load_mat("/oht_cfrp_00.tiff");
    REQUIRE(!ref.empty());
    const FullFieldParams p = oht_params();
    const int cap = capacity_for(p);

    IncrementalSession session;
    REQUIRE(session.init(ref, cv::Mat(), IncrementalParams{p, 1}) == 0);
    ReferenceCache fixed;
    fixed.set_from_gray(ref, cv::Mat());

    for (const Rung &rung : kLadder) {
        const cv::Mat def = load_mat(rung.image);
        REQUIRE(!def.empty());
        std::vector<Gold> gold;
        REQUIRE(load_gold(std::string(DICE_FIXTURES_DIR) + rung.gold, gold));

        std::vector<float> inc_out((size_t) cap, 0.0f), fix_out((size_t) cap, 0.0f);
        float metrics[23] = {};
        IncrementalFrameInfo info;
        const int n_inc = session.process_frame(def, inc_out.data(), cap, metrics, 23,
                                                nullptr, &info);
        const int n_fix = run_full_field(fixed, def, cv::Mat(), p, fix_out.data(), cap,
                                         metrics, 23);
        REQUIRE(n_inc > 0);
        REQUIRE(n_fix > 0);

        const Agreement inc = compare(inc_out, n_inc, gold);
        const Agreement fix = compare(fix_out, n_fix, gold);
        std::printf("  DiceIncremental %s: tracked %d/%d\n"
                    "     incremental: %d/%zu (%.0f%%)  mean du=%+.4f dv=%+.4f"
                    "  rms du=%.4f dv=%.4f  max|d|=%.4f\n"
                    "     fixed ref  : %d/%zu (%.0f%%)  mean du=%+.4f dv=%+.4f"
                    "  rms du=%.4f dv=%.4f  max|d|=%.4f\n",
                    rung.gold, info.tracked_points, info.frame0_points,
                    inc.compared, inc.gold, inc.frac() * 100.0, inc.mean_u, inc.mean_v,
                    inc.rms_u, inc.rms_v, inc.max_d,
                    fix.compared, fix.gold, fix.frac() * 100.0, fix.mean_u, fix.mean_v,
                    fix.rms_u, fix.rms_v, fix.max_d);

        CHECK(inc.frac() >= INC_MIN_FRACTION);
        CHECK(inc.rms_u < INC_RMS_TOL);
        CHECK(inc.rms_v < INC_RMS_TOL);
    }
}

#endif // DIC_HAVE_OPENCV
