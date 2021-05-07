#include <iostream>
#include <limits>
#include <cassert>
#include <cstring>
#include <chrono>

#include <ultimaille/all.h>

using namespace UM;

double triangle_area_2d(vec2 a, vec2 b, vec2 c) {
    return .5*((b.y-a.y)*(b.x+a.x) + (c.y-b.y)*(c.x+b.x) + (a.y-c.y)*(a.x+c.x));
}

double triangle_aspect_ratio_2d(vec2 a, vec2 b, vec2 c) {
    double l1 = (b-a).norm();
    double l2 = (c-b).norm();
    double l3 = (a-c).norm();
    double lmax = std::max(l1, std::max(l2, l3));
    return lmax*(l1+l2+l3)/(4.*std::sqrt(3.)*triangle_area_2d(a, b, c));
}

inline double chi(double eps, double det) {
    if (det>0)
        return (det + std::sqrt(eps*eps + det*det))*.5;
    return .5*eps*eps / (std::sqrt(eps*eps + det*det) - det);
}

inline double chi_deriv(double eps, double det) {
    return .5+det/(2.*std::sqrt(eps*eps + det*det));
}

struct Untangle2D {
    Untangle2D(Triangles &mesh) : m(mesh), X(m.nverts()*2), lock(m.points), ref_tri(m), J(m), K(m), det(m), area(m) {
        for (int t : facet_iter(m)) {
            area[t] = m.util.unsigned_area(t);
            vec2 A,B,C;
            m.util.project(t, A, B, C);

            double ar = triangle_aspect_ratio_2d(A, B, C);
            if (ar>10) { // if the aspect ratio is bad, assign an equilateral reference triangle
                double a = ((B-A).norm() + (C-B).norm() + (A-C).norm())/3.; // edge length is the average of the original triangle
                area[t] = sqrt(3.)/4.*a*a;
                A = {0., 0.};
                B = {a, 0.};
                C = {a/2., std::sqrt(3.)/2.*a};
            }

            mat<2,2> ST = {{B-A, C-A}};
            ref_tri[t] = mat<3,2>{{ {-1,-1},{1,0},{0,1} }}*ST.invert_transpose();
        }
    }

    void lock_boundary_verts() {
        SurfaceConnectivity fec(m);
        for (int v : vert_iter(m))
            lock[v] = fec.is_boundary_vert(v);
    }

    void evaluate_jacobian(const std::vector<double> &X) {
        detmin = std::numeric_limits<double>::max();
        ninverted = 0;
#pragma omp parallel for reduction(min:detmin) reduction(+:ninverted)
        for (int t=0; t<m.nfacets(); t++) {
            mat<2,2> &J = this->J[t];
            J = {};
            for (int i=0; i<3; i++)
                for (int d : range(2))
                    J[d] += ref_tri[t][i]*X[2*m.vert(t,i) + d];
            this->K[t] = { {{ +J[1].y, -J[1].x }, { -J[0].y, +J[0].x }} };  // dual basis
            det[t] = J.det();
            detmin = std::min(detmin, det[t]);
            ninverted += (det[t]<=0);
        }
    }

    bool go() {
        std::vector<SpinLock> spin_locks(X.size());
        eps = 1;
        evaluate_jacobian(X);
        if (debug>0) std::cerr <<  "number of inverted elements: " << ninverted << std::endl;
        for (int iter=0; iter<maxiter; iter++) {
            if (debug>0) std::cerr << "iteration #" << iter << std::endl;
            const LBFGS_Optimizer::func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
                std::fill(G.begin(), G.end(), 0);
                F = 0;
                evaluate_jacobian(X);
//#pragma omp parallel for reduction(vec_double_plus:G) reduction(+:F)
#pragma omp parallel for reduction(+:F)
                for (int t=0; t<m.nfacets(); t++) {
                    double c1 = chi(eps, det[t]);
                    double c2 = chi_deriv(eps, det[t]);

                    double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1])/c1;
                    double g = (1+det[t]*det[t])/c1;
                    F += ((1-theta)*f + theta*g)*area[t];

                    for (int dim : range(2)) {
                        vec2 a = J[t][dim]; // tangent basis
                        vec2 b = K[t][dim]; // dual basis
                        vec2 dfda = (a*2. - b*f*c2)/c1;
                        vec2 dgda = b*(2*det[t]-g*c2)/c1;
                        for (int i=0; i<3; i++) {
                            int v = m.vert(t,i);
                            if (lock[v]) continue;
                            spin_locks[v*2+dim].lock();
                            G[v*2+dim] += ((dfda*(1.-theta) + dgda*theta)*ref_tri[t][i])*area[t];
                            spin_locks[v*2+dim].unlock();
                        }
                    }
                }
//              double GG = std::sqrt(std::transform_reduce(G.begin(), G.end(), G.begin(), 0.));
//              double XX = std::sqrt(std::transform_reduce(X.begin(), X.end(), X.begin(), 0.));
//              std::cerr << F << " " << ninverted << " " << GG/std::max(1., XX) << std::endl;
            };

            double E_prev, E;
            std::vector<double> trash(X.size());
            func(X, E_prev, trash);

            LBFGS_Optimizer opt(func);
            opt.gtol = bfgs_threshold;
            opt.maxiter = bfgs_maxiter;
            opt.run(X);

            func(X, E, trash);
            if (debug>0) std::cerr << "E: " << E << " eps: " << eps << " detmin: " << detmin << " ninv: " << ninverted << std::endl;

#if 0
            double sigma = std::max(1.-E/E_prev, 1e-1);
            if (detmin>=0)
                eps *= (1-sigma);
            else
                eps *= 1 - (sigma*std::sqrt(detmin*detmin + eps*eps))/(std::abs(detmin) + std::sqrt(detmin*detmin + eps*eps));

#else
            double sigma = std::max(1.-E/E_prev, 1e-1);
            double mu = (1-sigma)*chi(eps, detmin);
            if (detmin<mu)
                eps = std::max(1e-9, 2*std::sqrt(mu*(mu-detmin)));
            else eps = 1e-9;
#endif

            if  (detmin>0 && std::abs(E_prev - E)/E<1e-5) break;
        }
        return !ninverted;
    }

    ////////////////////////////////
    // Untangle2D state variables //
    ////////////////////////////////

    // optimization input parameters
    Triangles &m;           // the mesh to optimize
    double theta = 1./128.; // the energy is (1-theta)*(shape energy) + theta*(area energy)
    int maxiter = 10000;    // max number of outer iterations
    double bfgs_threshold = 1e-4;
    int bfgs_maxiter = 30000; // max number of inner iterations
    int debug = 1;          // verbose level

    // optimization state variables

    std::vector<double> X;     // current geometry
    PointAttribute<bool> lock; // currently lock = boundary vertices
    FacetAttribute<mat<3,2>> ref_tri;
    FacetAttribute<mat<2,2>> J; // per-tet Jacobian matrix = [[JX.x JX.y, JX.z], [JY.x, JY.y, JY.z], [JZ.x, JZ.y, JZ.z]]
    FacetAttribute<mat<2,2>> K; // per-tet dual basis: det J = dot J[i] * K[i]
    FacetAttribute<double> det; // per-tet determinant of the Jacobian matrix
    FacetAttribute<double> area; // reference area
    double eps;       // regularization parameter, depends on min(jacobian)

    double detmin;    // min(jacobian) over all tetrahedra
    int ninverted; // number of inverted tetrahedra
};

int main(int argc, char** argv) {
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " model.mesh [result.mesh]" << std::endl;
        return 1;
    }

    std::string res_filename = "result.mesh";
    if (3<=argc) {
        res_filename = std::string(argv[2]);
    }

    Triangles m;
    SurfaceAttributes attr = read_by_extension(argv[1], m);
    std::cerr << "Untangling " << argv[1] << "," << m.nverts() << "," << std::endl;
    PointAttribute<vec2> tex_coord("tex_coord", attr, m);

    vec2 bbmin, bbmax; // these are used to undo the scaling we apply to the model
    const double boxsize = 10.;
    { // scale the target domain for better numerical stability
        bbmin = bbmax = tex_coord[0];
        for (int v : vert_iter(m)) {
            for (int d : range(2)) {
                bbmin[d] = std::min(bbmin[d], tex_coord[v][d]);
                bbmax[d] = std::max(bbmax[d], tex_coord[v][d]);
            }
        }
        double maxside = std::max(bbmax.x-bbmin.x, bbmax.y-bbmin.y);
        for (int v : vert_iter(m))
            tex_coord[v] = (tex_coord[v] - (bbmax+bbmin)/2.)*boxsize/maxside + vec2(1,1)*boxsize/2.;
    }

//    { // scale the input geometry to have the same area as the target domain
        double target_area = 0;
        for (int t : facet_iter(m)) {
            vec2 a = tex_coord[m.vert(t, 0)];
            vec2 b = tex_coord[m.vert(t, 1)];
            vec2 c = tex_coord[m.vert(t, 2)];
            target_area += triangle_area_2d(a, b, c);
        }
        um_assert(target_area>0); // ascertain mesh requirements
        double source_area = 0;
        for (int t : facet_iter(m))
            source_area += m.util.unsigned_area(t);
        for (vec3 &p : m.points)
            p *= std::sqrt(target_area/source_area);
//    }

    Untangle2D opt(m);

#if 0
    for (int t : facet_iter(m)) {
        opt.area[t] = target_area/m.nfacets();
        double a =  sqrt(opt.area[t]*4./sqrt(3.));
        vec2 A = {0., 0.};
        vec2 B = {a, 0.};
        vec2 C = {a/2., std::sqrt(3.)/2.*a};
        mat<2,2> ST = {{B-A, C-A}};
        opt.ref_tri[t] = mat<3,2>{{ {-1,-1},{1,0},{0,1} }}*ST.invert_transpose();
    }
#endif

    for (int v : vert_iter(m))
        for (int d : range(2))
            opt.X[2*v+d] = tex_coord[v][d];

    opt.lock_boundary_verts();

    auto t1 = std::chrono::high_resolution_clock::now();
    bool success = opt.go();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = t2 - t1;

    if (success)
        std::cerr << "SUCCESS; running time: " << time.count() << " s; min det J = " << opt.detmin << std::endl;
    else
        std::cerr << "FAIL TO UNTANGLE!" << std::endl;

    for (int v : vert_iter(m)) {
        for (int d : range(2))
            m.points[v][d] = opt.X[2*v+d];
        m.points[v].z = 0;
    }

    { // restore scale
        double maxside = std::max(bbmax.x-bbmin.x, bbmax.y-bbmin.y);
        for (vec3 &p : m.points)
            p = (p - vec3(1,1,1)*boxsize/2)/boxsize*maxside + (vec3(bbmax.x, bbmax.y, 0)+vec3(bbmin.x, bbmin.y, 0))/2.;
    }

    write_by_extension(res_filename, m, SurfaceAttributes{ { {"selection", opt.lock.ptr} }, { {"det", opt.det.ptr} }, {} });
    return 0;
}

