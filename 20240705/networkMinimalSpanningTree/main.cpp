#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>

using namespace std;

struct Computer {
    string name;
    string location;
};

struct Edge {
    int from, to, weight;
};

vector<Computer> computers;
vector<vector<int> > network;

void readComputerInfo(const string& filename) {
    ifstream file(filename.c_str());
    int n;
    file >> n;
    computers.resize(n);
    for (int i = 0; i < n; i++) {
        file >> computers[i].name >> computers[i].location;
    }
}

void readNetworkInfo(const string& filename) {
    ifstream file(filename.c_str());
    int n = computers.size();
    network.resize(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file >> network[i][j];
        }
    }
}

vector<Edge> prim() {
    int n = network.size();
    vector<bool> visited(n, false);
    vector<Edge> result;
    visited[0] = true;
    
    for (int i = 0; i < n - 1; i++) {
        Edge minEdge = {-1, -1, numeric_limits<int>::max()};
        for (int j = 0; j < n; j++) {
            if (visited[j]) {
                for (int k = 0; k < n; k++) {
                    if (!visited[k] && network[j][k] > 0 && network[j][k] < minEdge.weight) {
                        minEdge = (Edge){j, k, network[j][k]};
                    }
                }
            }
        }
        visited[minEdge.to] = true;
        result.push_back(minEdge);
    }
    return result;
}

int find(vector<int>& parent, int x) {
    if (parent[x] != x) {
        parent[x] = find(parent, parent[x]);
    }
    return parent[x];
}

void unite(vector<int>& parent, int x, int y) {
    parent[find(parent, x)] = find(parent, y);
}

vector<Edge> kruskal() {
    int n = network.size();
    vector<Edge> edges;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (network[i][j] > 0) {
                edges.push_back((Edge){i, j, network[i][j]});
            }
        }
    }
    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight;
    });
    
    vector<int> parent(n);
    for (int i = 0; i < n; i++) parent[i] = i;
    
    vector<Edge> result;
    for (size_t i = 0; i < edges.size(); i++) {
        const Edge& edge = edges[i];
        if (find(parent, edge.from) != find(parent, edge.to)) {
            unite(parent, edge.from, edge.to);
            result.push_back(edge);
        }
    }
    return result;
}

void writeResult(const string& filename) {
    ofstream file(filename.c_str());
    
    // 写入计算机信息
    file << computers.size() << endl;
    for (size_t i = 0; i < computers.size(); i++) {
        const Computer& comp = computers[i];
        file << comp.name << " " << comp.location << endl;
    }
    
    // 写入原始网络信息
    for (size_t i = 0; i < network.size(); i++) {
        const vector<int>& row = network[i];
        for (size_t j = 0; j < row.size(); j++) {
            file << row[j] << " ";
        }
        file << endl;
    }
    
    // 写入Prim算法结果
    vector<Edge> primResult = prim();
    int primTotalWeight = 0;
    file << "Prim" << endl;
    for (size_t i = 0; i < primResult.size(); i++) {
        const Edge& edge = primResult[i];
        file << edge.from << " " << edge.to << " " << edge.weight << endl;
        primTotalWeight += edge.weight;
    }
    file << "PrimTotalWeight " << primTotalWeight << endl;
    
    // 写入Kruskal算法结果
    vector<Edge> kruskalResult = kruskal();
    int kruskalTotalWeight = 0;
    file << "Kruskal" << endl;
    for (size_t i = 0; i < kruskalResult.size(); i++) {
        const Edge& edge = kruskalResult[i];
        file << edge.from << " " << edge.to << " " << edge.weight << endl;
        kruskalTotalWeight += edge.weight;
    }
    file << "KruskalTotalWeight " << kruskalTotalWeight << endl;
}

int main() {
    cout << "Output is in final.txt. Use HTML file to visualize it." <<endl;
    readComputerInfo("computer.txt");
    readNetworkInfo("network.txt");
    writeResult("final.txt");
    return 0;
}

