/*
    aabb.h -- functionality for creating adjacency matrices together with
              uniform or cotangent weights. Also contains data structures
              used to store integer variables.

    This file is part of the implementation of

        Instant Field-Aligned Meshes
        Wenzel Jakob, Daniele Panozzo, Marco Tarini, and Olga Sorkine-Hornung
        In ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2015)

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "adjacency.h"

namespace InstantMeshes
{

typedef Link** AdjacencyMatrixRef;

extern AdjacencyMatrixRef generate_adjacency_matrix_uniform(
    const MatrixXu &F, const VectorXu &V2E,
    const VectorXu &E2E, const VectorXb &nonManifold,
    const ProgressCallback &progress = ProgressCallback());

extern AdjacencyMatrixRef generate_adjacency_matrix_cotan(
    const MatrixXu &F, const MatrixXf &V, const VectorXu &V2E,
    const VectorXu &E2E, const VectorXb &nonManifold,
    const ProgressCallback &progress = ProgressCallback());

inline Link &search_adjacency(AdjacencyMatrixRef &adj, uint32_t i, uint32_t j) {
    for (Link* l = adj[i]; l != adj[i+1]; ++l)
        if (l->id == j)
            return *l;
    throw std::runtime_error("search_adjacency: failure!");
}

class BVH;
struct MeshStats;

extern AdjacencyMatrixRef generate_adjacency_matrix_pointcloud(
    MatrixXf &V, MatrixXf &N, const BVH *bvh, MeshStats &stats,
    uint32_t knn_points, bool deterministic = false,
    const ProgressCallback &progress = ProgressCallback());
}