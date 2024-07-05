#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

struct Computer {
	string name;
	string location;
};

struct Edge {
	int from, to, weight;
};

vector<Computer> computers;
vector<vector<int>> network;

void readComputerInfo(const string& filename) {
	ifstream file(filename);
	int n;
	file >> n;
	computers.resize(n);
	for (int i = 0; i < n; i++) {
		file >> computers[i].name >> computers[i].location;
	}
}

void readNetworkInfo(const string& filename) {
	ifstream file(filename);
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
						minEdge = {j, k, network[j][k]};
					}
				}
			}
		}
		visited[minEdge.to] = true;
		result.push_back(minEdge);
	}
	return result;
}

vector<Edge> kruskal() {
	int n = network.size();
	vector<Edge> edges;
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			if (network[i][j] > 0) {
				edges.push_back({i, j, network[i][j]});
			}
		}
	}
	sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
		return a.weight < b.weight;
	});
	
	vector<int> parent(n);
	for (int i = 0; i < n; i++) parent[i] = i;
	
	std::function<int(int)> find = [&](int x) {
		if (parent[x] != x) parent[x] = find(parent[x]);
		return parent[x];
	};
	
	auto unite = [&](int x, int y) {
		parent[find(x)] = find(y);
	};
	
	vector<Edge> result;
	for (const auto& edge : edges) {
		if (find(edge.from) != find(edge.to)) {
			unite(edge.from, edge.to);
			result.push_back(edge);
		}
	}
	return result;
}

void writeResult(const string& filename) {
	ofstream file(filename);
	
	// 写入计算机信息
	file << computers.size() << endl;
	for (const auto& comp : computers) {
		file << comp.name << " " << comp.location << endl;
	}
	
	// 写入原始网络信息
	for (const auto& row : network) {
		for (int weight : row) {
			file << weight << " ";
		}
		file << endl;
	}
	
	// 写入Prim算法结果
	auto primResult = prim();
	int primTotalWeight = 0;
	file << "Prim" << endl;
	for (const auto& edge : primResult) {
		file << edge.from << " " << edge.to << " " << edge.weight << endl;
		primTotalWeight += edge.weight;
	}
	file << "PrimTotalWeight " << primTotalWeight << endl;
	
	// 写入Kruskal算法结果
	auto kruskalResult = kruskal();
	int kruskalTotalWeight = 0;
	file << "Kruskal" << endl;
	for (const auto& edge : kruskalResult) {
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
