import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score
from sklearn.calibration import CalibrationDisplay
import shap

if platform.system() == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=plt.rcParams['font.sans-serif'][0])

# 1. 数据预处理与特征工程
df = pd.read_csv('cardio_train.csv', sep=';')
df.drop('id', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df = df[(df['ap_hi'] >= 90) & (df['ap_hi'] <= 200) & (df['ap_lo'] >= 60) & (df['ap_lo'] <= 130)]
df = df[df['ap_hi'] > df['ap_lo']]

df['age_years'] = (df['age'] / 365).round().astype(int)
df.drop('age', axis=1, inplace=True)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['map'] = df['ap_lo'] + (df['ap_hi'] - df['ap_lo']) / 3
df['unhealthy_lifestyle'] = df['smoke'] + df['alco'] + (1 - df['active'])

X = df.drop('cardio', axis=1)
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('体检数据特征相关性热力图', fontsize=16)
plt.tight_layout()
plt.show()

# 2. 多模型对比与评估调优
pipeline_lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42, max_iter=1000))])
pipeline_lr.fit(X_train, y_train)
lr_pred = pipeline_lr.predict(X_test)
print(f" -> [逻辑回归] 准确率 (Accuracy): {accuracy_score(y_test, lr_pred):.4f} | F1 分数: {f1_score(y_test, lr_pred):.4f}")

pipeline_rf = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))])
param_grid = {'rf__n_estimators': [100, 200], 'rf__max_depth': [10, 15]}
grid_search = GridSearchCV(pipeline_rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
rf_pred = best_rf_model.predict(X_test)
print(f" -> [随机森林] 准确率 (Accuracy): {accuracy_score(y_test, rf_pred):.4f} | F1 分数: {f1_score(y_test, rf_pred):.4f}")

models = {'逻辑回归': pipeline_lr, '随机森林': best_rf_model}

# 3. 临床可靠性检验
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

rf_pred = best_rf_model.predict(X_test)
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('随机森林 - 测试集混淆矩阵')

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
    CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, ax=axes[2], name=name)

axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlabel('假阳性率 (FPR)')
axes[1].set_ylabel('真阳性率 (TPR)')
axes[1].set_title('不同模型 ROC 曲线对比')
axes[1].legend(loc="lower right")

axes[2].set_title('模型校准曲线')
plt.tight_layout()
plt.show()

# 4. 风险解释与可视化
scaler = best_rf_model.named_steps['scaler']
X_test_sample = X_test.sample(n=1000, random_state=42)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_sample), columns=X.columns)

explainer = shap.TreeExplainer(best_rf_model.named_steps['rf'])
shap_values = explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):
    shap_vals_positive = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_vals_positive = shap_values[:, :, 1]
else:
    shap_vals_positive = shap_values

plt.close('all')
fig_shap = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_positive, X_test_scaled, show=False)
plt.title('SHAP 特征影响蜂窝图', fontsize=16, pad=15)
plt.tight_layout()
plt.show()


# 5. 医学健康提示
def launch_gui(pipeline, feature_cols):
    root = tk.Tk()
    root.title("心血管疾病风险 AI 辅助分析系统")
    root.geometry("650x850")
    root.configure(bg="#f0f5f9")

    style = ttk.Style()
    style.configure("TLabel", background="#f0f5f9", font=("微软雅黑", 10))

    header = tk.Frame(root, bg="#005088", height=80)
    header.pack(fill="x")
    tk.Label(header, text="心血管疾病风险 AI 辅助分析系统", bg="#005088", fg="white", font=("微软雅黑", 16, "bold"),
             pady=20).pack()

    main_frame = tk.Frame(root, bg="#f0f5f9", padx=30, pady=20)
    main_frame.pack(fill="both", expand=True)

    fields = [("年龄 (岁):", "age"), ("身高 (cm):", "height"), ("体重 (kg):", "weight"), ("收缩压 (高压):", "ap_hi"),
              ("舒张压 (低压):", "ap_lo")]
    entries = {}
    for i, (label_text, key) in enumerate(fields):
        tk.Label(main_frame, text=label_text).grid(row=i, column=0, sticky="w", pady=8)
        entry = ttk.Entry(main_frame, width=30)
        entry.grid(row=i, column=1, pady=8, padx=10)
        entries[key] = entry

    combos = [("性别:", "gender", ["1: 女性", "2: 男性"]),
              ("胆固醇:", "cholesterol", ["1: 正常", "2: 偏高", "3: 极高"]),
              ("血糖:", "gluc", ["1: 正常", "2: 偏高", "3: 极高"]), ("是否吸烟:", "smoke", ["0: 否", "1: 是"]),
              ("是否饮酒:", "alco", ["0: 否", "1: 是"]), ("是否运动:", "active", ["0: 否", "1: 是"])]
    combo_widgets = {}
    for i, (label_text, key, opts) in enumerate(combos):
        tk.Label(main_frame, text=label_text).grid(row=len(fields) + i, column=0, sticky="w", pady=8)
        cb = ttk.Combobox(main_frame, values=opts, width=28, state="readonly")
        cb.current(0)
        cb.grid(row=len(fields) + i, column=1, pady=8, padx=10)
        combo_widgets[key] = cb

    res_text = tk.Text(root, height=12, font=("微软雅黑", 10), bg="#ffffff", padx=10, pady=10)
    res_text.pack(padx=30, pady=10, fill="x")

    def run_prediction():
        try:
            age_val = float(entries['age'].get())
            height_val = float(entries['height'].get())
            weight_val = float(entries['weight'].get())
            ap_hi_val = float(entries['ap_hi'].get())
            ap_lo_val = float(entries['ap_lo'].get())

            if ap_lo_val >= ap_hi_val:
                messagebox.showwarning("数据异常拦截",
                                       "医学常识错误：舒张压(低压)不能高于或等于收缩压(高压)！\n请核对后重新输入。")
                return
            if not (50 <= ap_hi_val <= 300) or not (30 <= ap_lo_val <= 200):
                messagebox.showwarning("数据异常拦截", "血压数值超出人类生理极限！\n请核对是否多输或漏输了数字。")
                return
            if not (10 <= age_val <= 120):
                messagebox.showwarning("数据异常拦截", "年龄输入不符合常规范围（10-120岁）！")
                return
            if not (50 <= height_val <= 250):
                messagebox.showwarning("数据异常拦截", "身高输入不符合常规范围（50-250cm）！")
                return
            if not (20 <= weight_val <= 300):
                messagebox.showwarning("数据异常拦截", "体重输入不符合常规范围（20-300kg）！")
                return

            data = {
                'gender': int(combo_widgets['gender'].get().split(':')[0]),
                'height': height_val, 'weight': weight_val,
                'ap_hi': ap_hi_val, 'ap_lo': ap_lo_val,
                'cholesterol': int(combo_widgets['cholesterol'].get().split(':')[0]),
                'gluc': int(combo_widgets['gluc'].get().split(':')[0]),
                'smoke': int(combo_widgets['smoke'].get().split(':')[0]),
                'alco': int(combo_widgets['alco'].get().split(':')[0]),
                'active': int(combo_widgets['active'].get().split(':')[0]),
                'age_years': age_val
            }

            data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
            data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
            data['map'] = data['ap_lo'] + (data['ap_hi'] - data['ap_lo']) / 3
            data['unhealthy_lifestyle'] = data['smoke'] + data['alco'] + (1 - data['active'])

            input_df = pd.DataFrame([data])[feature_cols]
            prob = pipeline.predict_proba(input_df)[0][1]

            res_text.delete(1.0, tk.END)
            res_text.insert(tk.END, f"--- AI 智能诊断评估报告 ---\n\n")
            res_text.insert(tk.END, f"心血管疾病风险预测概率: {prob * 100:.2f}%\n")
            res_text.insert(tk.END, f"风险等级: {'【高风险】' if prob >= 0.5 else '【低/中风险】'}\n")
            res_text.insert(tk.END, f"身体质量指数 (BMI): {data['bmi']:.2f}\n")
            res_text.insert(tk.END, f"-" * 40 + "\n健康建议:\n")

            if data['ap_hi'] >= 140 or data['ap_lo'] >= 90:
                res_text.insert(tk.END, "血压超标: 请规范低钠饮食，规律监测血压，建议心内科就医复查。\n")
            if data['bmi'] >= 24:
                res_text.insert(tk.END, "体重预警: BMI偏高，建议科学减脂，调整饮食结构。\n")
            if data['unhealthy_lifestyle'] > 0:
                res_text.insert(tk.END, "生活方式: 建议戒烟限酒，保证每周至少3次有氧运动。\n")
            if prob < 0.5 and data['ap_hi'] < 140:
                res_text.insert(tk.END, "指标良好: 请继续保持当前健康的生活作息。\n")

        except ValueError:
            messagebox.showerror("输入格式错误", "包含非法字符或为空！请确保所有输入框均填入有效的阿拉伯数字。")

    tk.Button(main_frame, text="开始 AI 智能分析", command=run_prediction, bg="#11caa0", fg="white", padx=40, pady=10,
              borderwidth=0).grid(row=len(fields) + len(combos), column=0, columnspan=2, pady=25)
    root.mainloop()


launch_gui(best_rf_model, X.columns)