import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.interpolate import interp1d

# 加载 .npz 文件
data_path = 'PEMS04.npz'
data = np.load(data_path)

# 加载流量数据
flow_data = data['data']

# 假设每个时间点代表5分钟，创建时间序列
num_days = flow_data.shape[0] // 288  # 计算数据中有多少个完整的天
selected_day_index = 39
print(f"选择天数:{selected_day_index}")
start_index = selected_day_index * 288  # 计算选择的天的起始索引

dates = pd.date_range('2024-01-01', periods=flow_data.shape[0], freq='5min')
selected_day_dates = dates[start_index:start_index + 288]  # 选择一天的日期

# 随机选择5个节点
selected_node_indices = [30, 25, 284]
print("Selected Node Indices:", selected_node_indices)

# 选择一天的数据
one_day_data = flow_data[start_index:start_index + 288, selected_node_indices, 0]

# 创建DataFrame
df = pd.DataFrame(one_day_data, index=selected_day_dates, columns=[f'Node {i+1}' for i in range(3)])

# 打印出对应的时间范围
print("Selected Time Range:")
print("Start Time:", selected_day_dates.min())
print("End Time:", selected_day_dates.max())

# 定义每个节点的偏移量，自定义偏移量
offsets = np.array([-100, 150, 400])  # 自定义偏移量

# 用户自定义每条图线的颜色
custom_colors = ['#FFA500', '#006400', '#ADD8E6']  # 您可以根据需要更改这些颜色

# 绘制图表
plt.figure(figsize=(6.5, 5))

# 创建插值函数
interpolators = {column: interp1d(mdates.date2num(df.index), df[column] + offsets[i], kind='linear') for i, column in enumerate(df.columns)}

# 计算三个等分点
num_tokens = 4
start_time = selected_day_dates.min() + pd.Timedelta(minutes=450)
end_time = selected_day_dates[-1] - pd.Timedelta(minutes=800)
time_range = end_time - start_time
token_intervals = np.linspace(0, time_range.total_seconds(), num_tokens + 1)
token_times = start_time + pd.to_timedelta(token_intervals, unit='s')

# 用于存储交点数据
intersection_points = []

for i, column in enumerate(df.columns):
    plt.plot(df.index, df[column] + offsets[i], label=column, color=custom_colors[i], linewidth=2)
    
    # 使用插值函数计算每个token时间点的确切y值
    for token_time in token_times[1:-1]:  # 去掉最后一个点，因为它与结束时间重叠
        y_value = interpolators[column](mdates.date2num(token_time))  # 插值计算y值
        
        # 标记交点
        plt.scatter(token_time, y_value, color='silver', zorder=5)  # 使用散点图标记
        # 可以选择性地添加文本标签
        # plt.text(token_time, y_value, f'({token_time:%H:%M}, {y_value:.1f})',
        #          fontsize=8, ha='right', va='bottom')
        
        # 存储交点数据
        intersection_points.append((token_time, y_value, column))

# 设置日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))

# 隐藏左轴和下轴的刻度线
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# 旋转日期标记，避免重叠
plt.gcf().autofmt_xdate()

# 加粗坐标轴并显示箭头
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)  # 加粗坐标轴
    spine.set_color('black')  # 设置坐标轴颜色为黑色

# 隐藏上和右坐标轴
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 添加箭头
plt.annotate('', xy=(1, 0), xytext=(0.9999, 0),
             arrowprops=dict(facecolor='black', shrink=0.5, width=1, headwidth=5),
             xycoords='axes fraction', textcoords='axes fraction')

plt.annotate('', xy=(0, 1), xytext=(0, 0.9999),
             arrowprops=dict(facecolor='black', shrink=0.5, width=1, headwidth=5),
             xycoords='axes fraction', textcoords='axes fraction')

# 添加标签
plt.text(-0.04, 0.99, 'Value', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', rotation=90, fontweight='bold')
plt.text(0.99, 0.04, 'Time', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', fontweight='bold')
plt.text(0.2, 0.93, 'Token', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='red')
plt.text(0.45, 0.93, 'Token', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='red')
plt.text(0.7, 0.93, 'Token', transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='red')

# 自定义竖线的长短
ymin = 0.2  # 竖线的起始位置（相对于y轴的百分比）
ymax = 0.8  # 竖线的结束位置（相对于y轴的百分比）

# 添加Token竖线
for token_time in token_times[1:-1]:  # 去掉最后一个点，因为它与结束时间重叠
    plt.axvline(x=token_time, color='red', linewidth=4, ymin=0.08, ymax=0.9)

# 限制x轴的范围，只显示选择的一天的数据
plt.xlim([selected_day_dates.min() + pd.Timedelta(minutes=450), selected_day_dates[-1] - pd.Timedelta(minutes=800)])
plt.ylim(0, 1000)

# 显示图表
# plt.legend()
plt.show()