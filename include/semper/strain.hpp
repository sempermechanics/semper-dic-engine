#ifndef SEMPER_STRAIN_HPP
#define SEMPER_STRAIN_HPP

#include <semper/types.hpp>
#include <vector>

namespace Semper {

    struct DisplacementField {
        int width = 0, height = 0, step = 0;
        std::vector<float> u, v; // 🚀 Changed to float
        std::vector<bool> valid;
    };

    struct StrainField {
        std::vector<float> exx, eyy, exy; // 🚀 Changed to float
    };

    class StrainCalculator {
    public:
        // Implements DICe's Standard VSG (Linear Least Squares Plane Fit)
        static StrainField compute_vsg_strain(const DisplacementField& disp, int window_pixels);

        /**
         * How many grid nodes a full VSG window covers at this step — the count of
         * (dx, dy) offsets whose physical distance falls inside the window radius.
         *
         * compute_vsg_strain needs at least 3 to fit a plane, so a (step,
         * window_pixels) pair scoring below 3 rejects *every* point no matter how
         * well the correlation went. It depends only on these two numbers, which is
         * why run_full_field can check it before solving anything. Exposed rather
         * than duplicated so the guard and the calculator cannot drift apart.
         */
        static int vsg_window_node_count(int step, int window_pixels);
    };

}
#endif