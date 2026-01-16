import pandas as pd
import numpy as np
from typing import Optional, Union


def get_top_correlations(
        df: pd.DataFrame,
        target_column: str,
        top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    计算 DataFrame 中指定目标列与其他数值列的 Pearson 相关性，
    并按绝对值降序返回结果 DataFrame。

    Args:
        df: 输入的 Pandas DataFrame。
        target_column: 要计算相关性的目标列名。
        top_n: (可选) 只返回相关性绝对值最高的 N 行。如果为 None，则返回所有行。

    Returns:
        一个 DataFrame，包含 'Correlated_Column' (列名),
        'True_Correlation' (真值), 'Absolute_Correlation' (绝对值)，
        并按绝对值降序排序。
    """

    # 1. 过滤出所有的数值列
    numeric_df = df.select_dtypes(include=np.number)

    # 检查目标列是否在数值列中
    if target_column not in numeric_df.columns:
        raise ValueError(f"目标列 '{target_column}' 不是数值类型，无法计算相关性。")

    # 2. 计算目标列与所有数值列的相关系数
    # pandas.DataFrame.corr() 会返回一个 DataFrame，我们只取目标列对应的 Series
    correlation_series = numeric_df.corr()[target_column]

    # 3. 排除目标列自身的相关系数（与自身相关性总是 1）
    correlation_series = correlation_series.drop(target_column, errors='ignore')

    # 4. 构建结果 DataFrame
    result_df = pd.DataFrame({
        'Correlated_Column': correlation_series.index,
        'True_Correlation': correlation_series.values
    })

    # 5. 计算相关系数的绝对值，并作为新列加入
    result_df['Absolute_Correlation'] = result_df['True_Correlation'].abs()

    # 6. 按绝对值降序排序
    result_df = result_df.sort_values(
        by='Absolute_Correlation',
        ascending=False
    ).reset_index(drop=True)

    # 7. 返回前 N 行（如果指定了 top_n）
    if top_n is not None and top_n > 0:
        return result_df.head(top_n)

    return result_df


