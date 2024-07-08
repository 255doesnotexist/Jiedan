#include <stdio.h>
#include <string.h>
#define MAXN 60
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main() {
	int M;
	int head[MAXN], tail[MAXN];
	int dp[MAXN][MAXN];
	
	// 读取输入
	scanf("%d", &M);
	for (int i = 0; i < M; i++) {
		scanf("%d", &head[i]);
	}
	for (int i = 0; i < M - 1; i++) {
		tail[i] = head[i + 1];
	}
	tail[M - 1] = head[0]; // 环形结构
	
	// 初始化DP数组
	memset(dp, 0, sizeof(dp));
	
	// 动态规划计算最大能量
	for (int length = 2; length <= M; length++) {  // 子区间长度
		for (int i = 0; i < M; i++) {  // 子区间起点
			int j = (i + length - 1) % M;  // 子区间终点
			for (int k = i; k != j; k = (k + 1) % M) {  // 划分点
				int temp = dp[i][k] + dp[(k + 1) % M][j] + head[i] * tail[k % M] * tail[j % M];
				dp[i][j] = MAX(dp[i][j], temp);
			}
		}
	}
	
	// 找到最大能量
	int max_energy = 0;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			max_energy = MAX(max_energy, dp[i][j]);
		}
	}
	
	// 输出结果
	printf("%d\n", max_energy);
	
	return 0;
}

