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

#include "common.h"
#include <cstdint>

namespace InstantMeshes
{

/* Stores integer jumps between nodes of the adjacency matrix */
struct IntegerVariable {
    unsigned short rot : 2;
    signed   short translate_u : 7;
    signed   short translate_v : 7;

    Vector2i shift() const { 
        return Vector2i(translate_u, translate_v);
    }

    void setShift(Vector2i &v) {
        translate_u = v.x();
        translate_v = v.y();
    }
};

/* Stores a weighted adjacency matrix entry together with integer variables */
struct Link {
    uint32_t id;
    float weight;
    union {
        IntegerVariable ivar[2];
        uint32_t ivar_uint32;
    };

    inline Link() { }
    inline Link(uint32_t id) : id(id), weight(1.0f), ivar_uint32(0u) { }
    inline Link(uint32_t id, float weight) : id(id), weight(weight), ivar_uint32(0u) { }

    inline bool operator<(const Link &link) const { return id < link.id; }
};

class BVH;
struct MeshStats;

class AdjacencyMatrix
{
public:
    AdjacencyMatrix(){}
    // AdjacencyMatrix(const VectorXu& neighborhoodSize);
    // AdjacencyMatrix(const std::vector<uint32_t>& neighborhoodSize);
    AdjacencyMatrix(
        const std::vector<std::vector<uint32_t>>& adj_id,
        const std::vector<std::vector<uint32_t>>& adj_ivar,
        const std::vector<std::vector<Float>>& adj_weight);

    AdjacencyMatrix(
        const MatrixXu &F,
        const VectorXu &V2E,
        const VectorXu &E2E,
        const VectorXb &nonManifold,
        const ProgressCallback &progress = ProgressCallback());

    AdjacencyMatrix(
        const MatrixXu &F,
        const MatrixXf &V,
        const VectorXu &V2E,
        const VectorXu &E2E,
        const VectorXb &nonManifold,
        const ProgressCallback &progress = ProgressCallback());

    AdjacencyMatrix(
        MatrixXf &V,
        MatrixXf &N,
        const BVH& bvh,
        MeshStats &stats,
        uint32_t knn_points,
        bool deterministic = false,
        const ProgressCallback &progress = ProgressCallback());

    AdjacencyMatrix(
        const AdjacencyMatrix adj,
        const MatrixXf &V,
        const MatrixXf &N,
        const VectorXf &areas,
        MatrixXf &V_p,
        MatrixXf &V_n,
        VectorXf &areas_p,
        MatrixXu &to_upper,
        VectorXu &to_lower,
        bool deterministic = false,
        const ProgressCallback& progress = ProgressCallback());

    Link*& operator[] (size_t index) { return _rows.at(index); }
    Link* operator[] (size_t index) const { return _rows.at(index); }

public:
    std::vector<Link> _links;
    std::vector<Link*> _rows;
};

inline Link &search_adjacency(AdjacencyMatrix &adj, uint32_t i, uint32_t j) {
    for (Link* l = adj[i]; l != adj[i+1]; ++l)
        if (l->id == j)
            return *l;
    throw std::runtime_error("search_adjacency: failure!");
}

}