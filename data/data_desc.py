import pandas as pd

DropColumns = ['BMGs', "Chemical composition", "cls_label"]
TargetColumns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)','yield(MPa)', 'Modulus (GPa)', 'Ε(%)']


def get_row_matrix(df: pd.DataFrame):
    max_values = df.max(axis=1)
    df['element_base'] = df.apply(
        lambda row: row[row == max_values[row.name]].index.tolist(),
        axis=1
    )
    # df_expanded = df.explode('element_base').reset_index(drop=True)
    return df

reg_data_path = "./ALL_data_grouped_processed.xlsx"
reg_data = pd.read_excel(reg_data_path, sheet_name="Sheet1")
# reg_data = reg_data[reg_data["cls_label"] == 1]
reg_data.describe(percentiles=[.25, .5, .75, .8]).to_excel("./ALL_data_grouped_processed_des.xlsx")


reg_data = reg_data[reg_data['cls_label'] == 1]
df_f = reg_data.drop(columns=DropColumns).drop(columns=TargetColumns)
df_f = get_row_matrix(df_f)
reg_data['element_base'] = df_f['element_base']
reg_data_expanded = reg_data.explode('element_base').reset_index(drop=True)

# 按照element_base列进行分组
grouped = reg_data_expanded.groupby('element_base')
grouped.get_group("Zr").describe().to_excel("./ALL_data_grouped_processed_des_Zr.xlsx")

# 打印'Ε(%)'下有值的列他的element_base的分布，只需要计算每一个element_base下的'Ε(%)'的数量   
print(grouped['Ε(%)'].count())


cls_data_path = "./ALL_data_cls.xlsx"
cls_data = pd.read_excel(cls_data_path, sheet_name="Sheet1")
# 统计Class列的分布, 打印每一个label的数量和他的比例，注意要两列，数量也要，比例也要
cls_count = cls_data["Class"].value_counts()
cls_count_ratio = cls_data["Class"].value_counts(normalize=True)
cls_count_df = pd.DataFrame({"count": cls_count, "ratio": cls_count_ratio})
cls_count_df.to_excel("./ALL_data_cls_des.xlsx")