import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

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
selected_day_dates = dates[start_index:start_index + 288 + 288]  # 选择一天的日期

# 随机选择5个节点
selected_node_indices = [8, 53, 241, 278, 284]
print("Selected Node Indices:", selected_node_indices)

# 选择一天的数据
one_day_data = flow_data[start_index:start_index + 288 + 288, selected_node_indices, 0]

# 创建DataFrame
df = pd.DataFrame(one_day_data, index=selected_day_dates, columns=[f'Node {i+1}' for i in range(5)])

# 打印出对应的时间范围
print("Selected Time Range:")
print("Start Time:", selected_day_dates.min())
print("End Time:", selected_day_dates.max())

# 绘制图表
plt.figure(figsize=(10, 5))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# 设置日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))

# 添加图例
plt.legend(loc='upper left')

# # 设置标题和标签
# plt.xlabel('Time of Day')
plt.ylabel('Traffic Flow')

# 限制x轴的范围，只显示选择的一天的数据，确保结束时间是第二天的0:00
plt.xlim([selected_day_dates.min(), selected_day_dates[-1] + pd.Timedelta(minutes=5)])
plt.grid(True, linestyle='--')

# 添加矩形方框和文本标签
morning_start1 = pd.Timestamp('2024-02-09 06:40')
morning_end1 = pd.Timestamp('2024-02-09 09:20')
evening_start1 = pd.Timestamp('2024-02-09 16:40')
evening_end1 = pd.Timestamp('2024-02-09 19:20')

morning_start2 = pd.Timestamp('2024-02-10 06:40')
morning_end2 = pd.Timestamp('2024-02-10 09:20')
evening_start2 = pd.Timestamp('2024-02-10 16:40')
evening_end2 = pd.Timestamp('2024-02-10 19:20')

mid_morning1 = morning_start1 + (morning_end1 - morning_start1) / 2
mid_evening1 = evening_start1 + (evening_end1 - evening_start1) / 2
mid_morning2 = morning_start2 + (morning_end2 - morning_start2) / 2
mid_evening2 = evening_start2 + (evening_end2 - evening_start2) / 2

plt.axvspan(morning_start1, morning_end1, facecolor='yellow', alpha=0.3)
plt.text(mdates.date2num(mid_morning1), plt.ylim()[0]+62 - (plt.ylim()[1] - plt.ylim()[0]) * 0.05, 'Morning', ha='center', va='top', fontsize=8, color='black')

plt.axvspan(evening_start1, evening_end1, facecolor='green', alpha=0.3)
plt.text(mdates.date2num(mid_evening1), plt.ylim()[0]+62 - (plt.ylim()[1] - plt.ylim()[0]) * 0.05, 'Evening', ha='center', va='top', fontsize=8, color='black')

plt.axvspan(morning_start2, morning_end2, facecolor='pink', alpha=0.3)
plt.text(mdates.date2num(mid_morning2), plt.ylim()[0]+62 - (plt.ylim()[1] - plt.ylim()[0]) * 0.05, 'Morning', ha='center', va='top', fontsize=8, color='black')

plt.axvspan(evening_start2, evening_end2, facecolor='gray', alpha=0.3)
plt.text(mdates.date2num(mid_evening2), plt.ylim()[0]+62 - (plt.ylim()[1] - plt.ylim()[0]) * 0.05, 'Evening', ha='center', va='top', fontsize=8, color='black')

plt.savefig('traffic_flow.png', dpi=600, bbox_inches='tight')

# 显示图表
plt.show()
