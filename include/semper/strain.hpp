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

        // Normalized median test (Westerweel & Scarano 2005) on u and v over
        // each point's 5x5 neighbours, repeated until no new point fails.
        // Rejected points drop out of later passes, so a cluster of wrong
        // matches is peeled from its edge inward. Clears valid[] at each
        // rejected point and returns how many were rejected.
        static int reject_displacement_outliers(DisplacementField& disp);
    };

}
#endif