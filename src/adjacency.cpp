/*
    aabb.cpp -- functionality for creating adjacency matrices together with
                uniform or cotangent weights. Also contains data structures
                used to store integer variables.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "adjacency.h"
#include "dedge.h"
#include "bvh.h"
#include "meshstats.h"
#include "dset.h"
#include <set>
#include <map>
#include <numeric>
#include <parallel_stable_sort.h>

namespace InstantMeshes
{

// Load
AdjacencyMatrix::AdjacencyMatrix(
    const std::vector<std::vector<uint32_t>>& adj_id,
    const std::vector<std::vector<uint32_t>>& adj_ivar,
    const std::vector<std::vector<Float>>& adj_weight)
{
    uint32_t linkCount = 0;

    if (adj_id.size() != adj_ivar.size() || adj_ivar.size() != adj_weight.size())
            throw std::runtime_error("Could not unserialize data");

    for (uint32_t j=0; j<adj_id.size(); ++j) {
        if (adj_id[j].size() != adj_ivar[j].size() || adj_ivar[j].size() != adj_weight[j].size())
            throw std::runtime_error("Could not unserialize data");
        linkCount += adj_id[j].size();
    }

    mRows.resize(adj_id.size() + 1);
    mLinks.resize(linkCount);

    mRows[0] = mLinks.data();
    for (uint32_t i=0; i < adj_id.size(); ++i)
    {
        mRows[i+1] = mRows[i] + adj_id[i].size();
    }

    for (uint32_t j=0; j<adj_id.size(); ++j)
    {
        for (uint32_t k=0; k<adj_id[j].size(); ++k)
        {
            mRows[j][k].id = adj_id[j][k];
            mRows[j][k].ivar_uint32 = adj_ivar[j][k];
            mRows[j][k].weight = adj_weight[j][k];
        }
    }
}

// Uniform
AdjacencyMatrix::AdjacencyMatrix(
    const MatrixXu& F,
    const VectorXu& V2E,
    const VectorXu& E2E,
    const VectorXb& nonManifold,
    const ProgressCallback& progress)
{
    std::vector<uint32_t> adjacencySizes(V2E.size());
    if (logger) *logger << "Generating adjacency matrix .. " << std::flush;
    Timer<> timer;

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V2E.size(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                uint32_t edge = V2E[i], stop = edge;
                if (nonManifold[i] || edge == INVALID) {
                    adjacencySizes[i] = 0;
                    continue;
                }
                uint32_t nNeighbors = 0;
                do {
                    uint32_t opp = E2E[edge];
                    if (opp == INVALID) {
                        nNeighbors += 2;
                        break;
                    }
                    edge = dedge_next_3(opp);
                    nNeighbors++;
                } while (edge != stop);
                adjacencySizes[i] = nNeighbors;
            }
            SHOW_PROGRESS_RANGE(range, V2E.size(), "Generating adjacency matrix (1/2)");
        }
    );

    const uint32_t linkCount = std::accumulate(adjacencySizes.begin(), adjacencySizes.end(), 0);

    mRows.resize(adjacencySizes.size() + 1);
    mLinks.resize(linkCount);

    mRows[0] = mLinks.data();
    for (uint32_t i=0; i < adjacencySizes.size(); ++i)
    {
        mRows[i+1] = mRows[i] + adjacencySizes[i];
    }

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V2E.size(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                uint32_t edge = V2E[i], stop = edge;
                if (nonManifold[i] || edge == INVALID)
                    continue;
                Link *ptr = mRows[i];

                int it = 0;
                do {
                    uint32_t base = edge % 3, f = edge / 3;
                    uint32_t opp = E2E[edge], next = dedge_next_3(opp);
                    if (it == 0)
                        *ptr++ = Link(F((base + 2)%3, f));
                    if (opp == INVALID || next != stop) {
                        *ptr++ = Link(F((base + 1)%3, f));
                        if (opp == INVALID)
                            break;
                    }
                    edge = next;
                    ++it;
                } while (edge != stop);
            }
            SHOW_PROGRESS_RANGE(range, V2E.size(), "Generating adjacency matrix (2/2)");
        }
    );

    if (logger) *logger << "done. (took " << timeString(timer.value()) << ")" << std::endl;
}

// Cotangent Laplacian
AdjacencyMatrix::AdjacencyMatrix(
    const MatrixXu& F,
    const MatrixXf& V,
    const VectorXu& V2E,
    const VectorXu& E2E,
    const VectorXb& nonManifold,
    const ProgressCallback& progress)
{
    std::vector<uint32_t> adjacencySizes(V2E.size());
    if (logger) *logger << "Computing cotangent Laplacian .. " << std::flush;
    Timer<> timer;

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V2E.size(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                uint32_t edge = V2E[i], stop = edge;
                if (nonManifold[i] || edge == INVALID) {
                    adjacencySizes[i] = 0;
                    continue;
                }
                uint32_t nNeighbors = 0;
                do {
                    uint32_t opp = E2E[edge];
                    if (opp == INVALID) {
                        nNeighbors += 2;
                        break;
                    }
                    edge = dedge_next_3(opp);
                    nNeighbors++;
                } while (edge != stop);
                adjacencySizes[i] = nNeighbors;
            }
            SHOW_PROGRESS_RANGE(range, V2E.size(), "Computing cotangent Laplacian (1/2)");
        }
    );

    const uint32_t linkCount = std::accumulate(adjacencySizes.begin(), adjacencySizes.end(), 0);

    mRows.resize(adjacencySizes.size() + 1);
    mLinks.resize(linkCount);

    mRows[0] = mLinks.data();
    for (uint32_t i=0; i < adjacencySizes.size(); ++i)
    {
        mRows[i+1] = mRows[i] + adjacencySizes[i];
    }

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t)V.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                uint32_t edge = V2E[i], stop = edge;
                if (nonManifold[i] || edge == INVALID)
                    continue;
                Link *ptr = mRows[i];

                int it = 0;
                do {
                    uint32_t f = edge / 3, curr_idx = edge % 3,
                             next_idx = (curr_idx + 1) % 3,
                             prev_idx = (curr_idx + 2) % 3;

                    if (it == 0) {
                        Vector3f p  = V.col(F(next_idx, f)),
                                 d0 = V.col(F(prev_idx, f)) - p,
                                 d1 = V.col(F(curr_idx, f)) - p;
                        Float cot_weight = 0.0f,
                              sin_alpha = d0.cross(d1).norm();
                        if (sin_alpha > RCPOVERFLOW)
                            cot_weight = d0.dot(d1) / sin_alpha;

                        uint32_t opp = E2E[dedge_prev_3(edge)];
                        if (opp != INVALID) {
                            uint32_t o_f = opp / 3, o_curr_idx = opp % 3,
                                     o_next_idx = (o_curr_idx + 1) % 3,
                                     o_prev_idx = (o_curr_idx + 2) % 3;
                            p  = V.col(F(o_prev_idx, o_f));
                            d0 = V.col(F(o_curr_idx, o_f)) - p;
                            d1 = V.col(F(o_next_idx, o_f)) - p;
                            sin_alpha = d0.cross(d1).norm();
                            if (sin_alpha > RCPOVERFLOW)
                                cot_weight += d0.dot(d1) / sin_alpha;
                        }

                        *ptr++ = Link(F(prev_idx, f), (float) cot_weight * 0.5f);
                    }
                    uint32_t opp = E2E[edge], next = dedge_next_3(opp);
                    if (opp == INVALID || next != stop) {
                        Vector3f p  = V.col(F(prev_idx, f)),
                                 d0 = V.col(F(curr_idx, f)) - p,
                                 d1 = V.col(F(next_idx, f)) - p;
                        Float cot_weight = 0.0f,
                              sin_alpha = d0.cross(d1).norm();
                        if (sin_alpha > RCPOVERFLOW)
                            cot_weight = d0.dot(d1) / sin_alpha;

                        if (opp != INVALID) {
                            uint32_t o_f = opp / 3, o_curr_idx = opp % 3,
                                     o_next_idx = (o_curr_idx + 1) % 3,
                                     o_prev_idx = (o_curr_idx + 2) % 3;
                            p  = V.col(F(o_prev_idx, o_f));
                            d0 = V.col(F(o_curr_idx, o_f)) - p;
                            d1 = V.col(F(o_next_idx, o_f)) - p;
                            sin_alpha = d0.cross(d1).norm();
                            if (sin_alpha > RCPOVERFLOW)
                                cot_weight += d0.dot(d1) / sin_alpha;
                        }

                        *ptr++ = Link(F(next_idx, f), (float) cot_weight * 0.5f);
                        if (opp == INVALID)
                            break;
                    }
                    edge = next;
                    ++it;
                } while (edge != stop);
            }
            SHOW_PROGRESS_RANGE(range, V.cols(), "Computing cotangent Laplacian (2/2)");
        }
    );
    if (logger) *logger << "done. (took " << timeString(timer.value()) << ")" << std::endl;
}

// Point cloud
AdjacencyMatrix::AdjacencyMatrix(
    MatrixXf &V,
    MatrixXf &N,
    const BVH& bvh,
    MeshStats &stats,
    uint32_t knn_points,
    bool deterministic,
    const ProgressCallback &progress)
{
    Timer<> timer;
    if (logger) *logger << "Generating adjacency matrix .. " << std::flush;

    stats.mAverageEdgeLength = bvh.diskRadius();
    const Float maxQueryRadius = bvh.diskRadius() * 3;

    std::vector<uint32_t> adjacencySets(V.cols() * knn_points);
    auto adj_sets = adjacencySets.data();

    DisjointSets dset(V.cols());
    std::vector<uint32_t> adjacencySizes(V.cols());
    //VectorXu adj_size(V.cols());
    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            std::vector<std::pair<Float, uint32_t>> result;
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                uint32_t *adj_set = adj_sets + (size_t) i * (size_t) knn_points;
                memset(adj_set, 0xFF, sizeof(uint32_t) * knn_points);
                Float radius = maxQueryRadius;
                bvh.findKNearest(V.col(i), N.col(i), knn_points, radius, result);
                uint32_t ctr = 0;
                for (auto k : result) {
                    if (k.second == i)
                        continue;
                    adj_set[ctr++] = k.second;
                    dset.unite(k.second, i);
                }
                adjacencySizes[i] = ctr;
            }
            SHOW_PROGRESS_RANGE(range, V.cols(), "Generating adjacency matrix");
        }
    );

    std::map<uint32_t, uint32_t> dset_size;
    for (uint32_t i=0; i<V.cols(); ++i) {
        dset_size[dset.find(i)]++;
        uint32_t *adj_set_i = adj_sets + (size_t) i * (size_t) knn_points;

        for (uint32_t j=0; j<knn_points; ++j) {
            uint32_t k = adj_set_i[j];
            if (k == INVALID)
                break;
            uint32_t *adj_set_k = adj_sets + (size_t) k * (size_t) knn_points;
            bool found = false;
            for (uint32_t l=0; l<knn_points; ++l) {
                uint32_t value = adj_set_k[l];
                if (value == i) { found = true; break; }
                if (value == INVALID) break;
            }
            if (!found)
                adjacencySizes[k]++;
        }
    }

    size_t linkCount = 0;
    for (uint32_t i=0; i<V.cols(); ++i)
    {
        const uint32_t dsetSize = dset_size[dset.find(i)];
        uint32_t& adjacencySize = adjacencySizes[i];
        if (dsetSize < V.cols() * 0.01f)
        {
            adjacencySize = INVALID;
            V.col(i) = Vector3f::Constant(1e6);
        }
        else
        {
            linkCount += adjacencySize;
        }
    }

    if (logger) *logger << "allocating " << memString(sizeof(Link) * linkCount) << " .. " << std::flush;

    mRows.resize(adjacencySizes.size() + 1);
    mLinks.resize(linkCount);
    mRows[0] = mLinks.data();

    for (uint32_t i=0; i < adjacencySizes.size(); ++i)
    {
        const uint32_t size = adjacencySizes[i];
        if (size == INVALID)
        {
            mRows[i+1] = mRows[i];
        }
        else
        {
            mRows[i+1] = mRows[i] + size;
        }
    }

    VectorXu adj_offset(V.cols());
    adj_offset.setZero();

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                uint32_t *adj_set_i = adj_sets + (size_t) i * (size_t) knn_points;
                if (adjacencySizes[i] == INVALID)
                    continue;

                for (uint32_t j=0; j<knn_points; ++j) {
                    uint32_t k = adj_set_i[j];
                    if (k == INVALID)
                        break;
                    mRows[i][atomicAdd(&adj_offset.coeffRef(i), 1)-1] = Link(k);

                    uint32_t *adj_set_k = adj_sets + (size_t) k * (size_t) knn_points;
                    bool found = false;
                    for (uint32_t l=0; l<knn_points; ++l) {
                        uint32_t value = adj_set_k[l];
                        if (value == i) { found = true; break; }
                        if (value == INVALID) break;
                    }
                    if (!found)
                        mRows[k][atomicAdd(&adj_offset.coeffRef(k), 1)-1] = Link(i);
                }
            }
        }
    );

    /* Use a heuristic to estimate some useful quantities for point clouds (this
       is a biased estimate due to the kNN queries, but it's convenient and
       reasonably accurate) */
    stats.mSurfaceArea = M_PI * stats.mAverageEdgeLength*stats.mAverageEdgeLength * 0.5f * V.cols();

    if (logger) *logger << "done. (took " << timeString(timer.value()) << ")" << std::endl;
}

AdjacencyMatrix::AdjacencyMatrix(
    const AdjacencyMatrix& adj,
    const MatrixXf &V,
    const MatrixXf &N,
    const VectorXf &A,
    MatrixXf& V_p,
    MatrixXf& N_p,
    VectorXf& A_p,
    MatrixXu& to_upper,
    VectorXu& to_lower,
    bool deterministic,
    const ProgressCallback &progress)
{
    struct Entry {
        uint32_t i, j;
        float order;
        inline Entry() { };
        inline Entry(uint32_t i, uint32_t j, float order) : i(i), j(j), order(order) { }
        inline bool operator<(const Entry &e) const { return order > e.order; }
    };

    uint32_t nLinks = adj[V.cols()] - adj[0];
    std::vector<Entry> entryBuffer(nLinks);
    Entry* entries = entryBuffer.data();
    Timer<> timer;
    if (logger) *logger << "  Collapsing .. " << std::flush;

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                uint32_t nNeighbors = adj[i + 1] - adj[i];
                uint32_t base = adj[i] - adj[0];
                for (uint32_t j = 0; j < nNeighbors; ++j) {
                    uint32_t k = adj[i][j].id;
                    Float dp = N.col(i).dot(N.col(k));
                    Float ratio = A[i]>A[k] ? (A[i]/A[k]) : (A[k]/A[i]);
                    entries[base + j] = Entry(i, k, dp * ratio);
                }
            }
            SHOW_PROGRESS_RANGE(range, V.cols(), "Downsampling graph (1/6)");
        }
    );

    if (progress)
        progress("Downsampling graph (2/6)", 0.0f);

    if (deterministic)
        pss::parallel_stable_sort(entries, entries + nLinks, std::less<Entry>());
    else
        tbb::parallel_sort(entries, entries + nLinks, std::less<Entry>());

    std::vector<bool> mergeFlag(V.cols(), false);

    uint32_t nCollapsed = 0;
    for (uint32_t i=0; i<nLinks; ++i) {
        const Entry &e = entries[i];
        if (mergeFlag[e.i] || mergeFlag[e.j])
            continue;
        mergeFlag[e.i] = mergeFlag[e.j] = true;
        entries[nCollapsed++] = entries[i];
    }
    uint32_t vertexCount = V.cols() - nCollapsed;

    /* Allocate memory for coarsened graph */
    V_p.resize(3, vertexCount);
    N_p.resize(3, vertexCount);
    A_p.resize(vertexCount);
    to_upper.resize(2, vertexCount);
    to_lower.resize(V.cols());

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) nCollapsed, GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                const Entry &e = entries[i];
                const Float area1 = A[e.i], area2 = A[e.j], surfaceArea = area1+area2;
                if (surfaceArea > RCPOVERFLOW)
                    V_p.col(i) = (V.col(e.i) * area1 + V.col(e.j) * area2) / surfaceArea;
                else
                    V_p.col(i) = (V.col(e.i) + V.col(e.j)) * 0.5f;
                Vector3f normal = N.col(e.i) * area1 + N.col(e.j) * area2;
                Float norm = normal.norm();
                N_p.col(i) = norm > RCPOVERFLOW ? Vector3f(normal / norm)
                                                : Vector3f::UnitX();
                A_p[i] = surfaceArea;
                to_upper.col(i) << e.i, e.j;
                to_lower[e.i] = i; to_lower[e.j] = i;
            }
            SHOW_PROGRESS_RANGE(range, nCollapsed, "Downsampling graph (3/6)");
        }
    );

    std::atomic<int> offset(nCollapsed);
    tbb::blocked_range<uint32_t> range(0u, (uint32_t) V.cols(), GRAIN_SIZE);

    auto copy_uncollapsed = [&](const tbb::blocked_range<uint32_t> &range) {
        for (uint32_t i = range.begin(); i != range.end(); ++i) {
            if (!mergeFlag[i]) {
                uint32_t idx = offset++;
                V_p.col(idx) = V.col(i);
                N_p.col(idx) = N.col(i);
                A_p[idx] = A[i];
                to_upper.col(idx) << i, INVALID;
                to_lower[i] = idx;
            }
        }
        SHOW_PROGRESS_RANGE(range, V.cols(), "Downsampling graph (4/6)");
    };

    if (deterministic)
        copy_uncollapsed(range);
    else
        tbb::parallel_for(range, copy_uncollapsed);

    std::vector<uint32_t> adjacencySizes(V_p.cols());

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V_p.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            std::vector<Link> scratch;
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                scratch.clear();

                for (int j=0; j<2; ++j) {
                    uint32_t upper = to_upper(j, i);
                    if (upper == INVALID)
                        continue;
                    for (Link *link = adj[upper]; link != adj[upper+1]; ++link)
                        scratch.push_back(Link(to_lower[link->id], link->weight));
                }

                std::sort(scratch.begin(), scratch.end());
                uint32_t id = INVALID, size = 0;
                for (const auto &link : scratch) {
                    if (id != link.id && link.id != i) {
                        id = link.id;
                        ++size;
                    }
                }
                adjacencySizes[i] = size;
            }
            SHOW_PROGRESS_RANGE(range, V_p.cols(), "Downsampling graph (5/6)");
        }
    );

    const uint32_t linkCount = std::accumulate(adjacencySizes.begin(), adjacencySizes.end(), 0);

    mRows.resize(adjacencySizes.size() + 1);
    mLinks.resize(linkCount);

    mRows[0] = mLinks.data();
    for (uint32_t i=0; i < adjacencySizes.size(); ++i)
    {
        mRows[i+1] = mRows[i] + adjacencySizes[i];
    }

    tbb::parallel_for(
        tbb::blocked_range<uint32_t>(0u, (uint32_t) V_p.cols(), GRAIN_SIZE),
        [&](const tbb::blocked_range<uint32_t> &range) {
            std::vector<Link> scratch;
            for (uint32_t i = range.begin(); i != range.end(); ++i) {
                scratch.clear();

                for (int j=0; j<2; ++j) {
                    uint32_t upper = to_upper(j, i);
                    if (upper == INVALID)
                        continue;
                    for (Link *link = adj[upper]; link != adj[upper+1]; ++link)
                        scratch.push_back(Link(to_lower[link->id], link->weight));
                }
                std::sort(scratch.begin(), scratch.end());
                Link *dest = mRows[i];
                uint32_t id = INVALID;
                for (const auto &link : scratch) {
                    if (link.id != i) {
                        if (id != link.id) {
                            *dest++ = link;
                            id = link.id;
                        } else {
                            dest[-1].weight += link.weight;
                        }
                    }
                }
            }
            SHOW_PROGRESS_RANGE(range, V_p.cols(), "Downsampling graph (6/6)");
        }
    );
    if (logger) *logger << "done. (" << V.cols() << " -> " << V_p.cols() << " vertices, took "
         << timeString(timer.value()) << ")" << std::endl;
}

}