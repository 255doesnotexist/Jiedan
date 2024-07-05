# %% [markdown]
# 1. 创建宿舍成员基本信息表(DataFrame)。

# %%
import pandas as pd

# 创建数据
data = {
    '编号': [1, 2, 3, 4, 5, 6],
    '姓名': ['郭康欣', '边水柔', '崔以墨', '耿雨菲', '郭硕', '占位'],
    '学号': ['2305014208', '2305014202', '2305014204', '2305014207', '2305014209', '2300000000'],
    '性别': ['女', '女', '女', '女', '女', '女'],
    '年龄': [19, 19, 19, 19, 19, 19],
    '出生年月日': ['2004-01-15', '2004-03-22', '2004-05-10', '2004-07-18', '2004-09-25', '2004-01-01'],
    '身高': [165, 160, 158, 162, 167, 160],
    '体重': [50, 48, 47, 52, 55, 60],
    '爱好': ['阅读', '跑步', '音乐', '绘画', '篮球', '占位'],
    '宿舍号': [101, 101, 101, 101, 101, 101]
}

# 创建DataFrame
df = pd.DataFrame(data)
display(df)

# %% [markdown]
# 2. 求该宿舍成员的身高、体重的均值、最大值、最小值、标准差等统计信息，并按出生年月日对该宿舍成员进行排序
# 

# %%
# 计算统计信息
height_stats = df['身高'].astype(float).describe()
weight_stats = df['体重'].astype(float).describe()

print("身高统计信息：")
display(height_stats)

print("体重统计信息：")
display(weight_stats)

# 按出生年月日排序
sorted_df = df.sort_values(by='出生年月日')
display(sorted_df)


# %% [markdown]
# 3. 创建2023-2024学年第一学期宿舍成员科目成绩信息表
# 

# %%
# 创建成绩数据
grades = {
    '编号': [1, 2, 3, 4, 5, 6],
    '姓名': ['郭康欣', '边水柔', '崔以墨', '耿雨菲', '郭硕', '占位'],
    '学号': ['2305014208', '2305014202', '2305014204', '2305014207', '2305014209', '2300000000'],
    '数学分析(一)': [85, 90, 78, 88, 92, 90],
    '解析几何': [88, 85, 80, 82, 89, 90],
    '大学计算机': [90, 92, 85, 87, 91, 90],
    '大学英语(1)': [87, 89, 83, 86, 88, 90],
    '思想道德修养与法律基础': [86, 88, 81, 84, 87, 90]
}

# 创建DataFrame
grades_df = pd.DataFrame(grades)
display(grades_df)


# %% [markdown]
# 4. 使用isnull()对宿舍成员成绩信息表进行缺失值检测

# %%
# 检测缺失值
missing_values = grades_df.isnull().sum()
print(missing_values)

# %% [markdown]
# 5. 宿舍成员科目成绩求和，并追加到表最后一列，并按成绩进行排序

# %%
# 计算总成绩
grades_df['总成绩'] = grades_df.iloc[:, 3:].sum(axis=1)

# 按总成绩排序
sorted_grades_df = grades_df.sort_values(by='总成绩', ascending=False)
display(sorted_grades_df)

# %% [markdown]
# 6. 用describe()方法对各科目成绩进行描述性统计
# 

# %%
# 描述性统计
stats = grades_df.describe()
display(stats)

# %% [markdown]
# 7. 使用Matplotlib库分别绘制数学分析（一）成绩的柱状图
# 

# %%
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签（中文乱码问题）

# 绘制柱状图
plt.bar(grades_df['姓名'], grades_df['数学分析(一)'])
plt.xlabel('姓名')
plt.ylabel('数学分析(一)成绩')
plt.title('数学分析(一)成绩分布')
plt.show()

# %% [markdown]
# 8. 使用Matplotlib库分别绘制数学分析（一）成绩的饼图
# 

# %%
# 绘制饼图
plt.pie(grades_df['数学分析(一)'], labels=grades_df['姓名'], autopct='%1.1f%%')
plt.title('数学分析(一)成绩分布')
plt.show()

# %% [markdown]
# 9. 用箱线图分别对数学分析（一）成绩进行异常值检测

# %%
# 绘制箱线图
plt.boxplot(grades_df['数学分析(一)'])
plt.xticks([1], ['数学分析(一)'])
plt.ylabel('成绩')
plt.title('数学分析(一)成绩箱线图')
plt.show()

# %% [markdown]
# 10. 按姓名和学号将宿舍成员基本信息表和成绩信息表进行合并
# 

# %%
# 合并数据
merged_df = pd.merge(df, grades_df, on=['姓名', '学号'])
display(merged_df)

# %%



