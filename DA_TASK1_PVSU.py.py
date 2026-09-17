'''
Анализ данных системы кондиционирования распределительного центра.

Полетаев В.И
Ютик С.И
                                (Ultimate Lizzard TM <3)
'''
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# missingno - библиотека для визуализации пропусков

try:
    import missingno as msno
    HAS_MISSINGNO = True
except ImportError:
    HAS_MISSINGNO = False

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

# 1-2. Загрузка исходного датасета с корректной обработкой формата

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        skiprows=[0],          # служебная строка с метаданными файла
        header=[0, 1, 2, 3],   
        encoding="utf-8",
        encoding_errors="ignore",
        low_memory=False,
    )
    return df


# 3. Очистка заголовков колонок

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Убираем незначащие уровни MultiIndex (Units, Digital, Sample Rate) -
    df = df.copy()
    df.columns = df.columns.droplevel([1, 2, 3])

    def clean_one(col: str) -> str:
        col = re.sub(r"\s+", " ", str(col)).strip()   # убираем лишние пробелы и табы
        col = col.replace("---", "").strip()            # убираем "---"
        parts = [p.strip() for p in col.split(":", 1)]
        if len(parts) == 2:
            return f"{parts[0]}: {parts[1]}"
        return col

    new_cols = [clean_one(c) for c in df.columns]
    new_cols[0] = "time"  # первая колонка - метка времени

    # На случай дублирующихся имён после очистки - делаем их уникальными
    seen = {}
    unique_cols = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c} ({seen[c]})")
        else:
            seen[c] = 0
            unique_cols.append(c)

    df.columns = unique_cols
    return df

# 5.datetime, индекс, сортировка
def set_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = df["time"].astype(str).str.strip()
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S  %d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.set_index("time")
    df = df.sort_index()
    return df


def get_state_columns(df: pd.DataFrame) -> list:
    #Колонки со статусом устройства (категориальные, 'EKC состояние')
    return [c for c in df.columns if "EKC состояние" in c]

def to_numeric_value_columns(df: pd.DataFrame, state_cols: list) -> pd.DataFrame:
    #Приводим все "непрерывные" колонки к числовому типу
    df = df.copy()
    value_cols = [c for c in df.columns if c not in state_cols]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data_example.csv"

    print(" ")
    raw = load_raw(path)
    print("3. Очистка заголовков колонок")
    print(" ")
    data = clean_columns(raw)
    print("Пример колонок после очистки:")
    print(list(data.columns[:6]))
    print("\n" + "=" * 80)
    print("5. Время -> datetime индекс, сортировка")
    print("=" * 80)
    data = set_time_index(data)
    print(f"Диапазон времени: {data.index.min()} .. {data.index.max()}")
    print(f"Индекс монотонно возрастает: {data.index.is_monotonic_increasing}")

    state_cols = get_state_columns(data)
    data = to_numeric_value_columns(data, state_cols)

    print("\n" + "=" * 80)
    print("4. Информация о датасете после очистки")
    print("=" * 80)
    data.info()

    print("\n" + "=" * 80)
    print("6. Количество пропусков по колонкам")
    print("=" * 80)
    na_counts = data.isna().sum()
    print(na_counts[na_counts > 0].sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(14, 6))
    if HAS_MISSINGNO:
        msno.matrix(data, ax=ax)
    else:
        sns.heatmap(data.isna(), cbar=False, yticklabels=False, ax=ax, cmap="viridis")
    ax.set_title("Карта пропусков по датасету")
    plt.tight_layout()
    plt.savefig("06_missing_values.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("7. Оптимизация типов данных (int / category)")
    print("=" * 80)
    mem_before = data.memory_usage(deep=True).sum()

    data_opt = data.copy()
    value_cols = [c for c in data_opt.columns if c not in state_cols]

    # float - целочисленный тип там, где значения фактически целые
    # (Int64 - nullable-тип, умеет хранить пропуски в отличие от обычного int)
    for c in value_cols:
        col = data_opt[c]
        non_na = col.dropna()
        if len(non_na) and (non_na == non_na.round()).all():
            data_opt[c] = col.astype("Int64")

    # категориальные признаки: EKC состояние, а также любые другие колонки,
    # у которых мало уникальных значений
    cat_candidates = list(state_cols) + [
        c for c in value_cols
        if data_opt[c].nunique(dropna=True) <= 10
    ]
    for c in cat_candidates:
        data_opt[c] = data_opt[c].astype("category")

    mem_after = data_opt.memory_usage(deep=True).sum()
    print(f"Память до оптимизации:  {mem_before / 1024 ** 2:.2f} МБ")
    print(f"Память после оптимизации: {mem_after / 1024 ** 2:.2f} МБ")
    print(f"Экономия: {(1 - mem_after / mem_before) * 100:.1f}%")
    # Важно: оригинальный датафрейм data не перезаписываем, работаем с data_opt

    print("\n" + "=" * 80)
    print("8. Статистика по параметру Sair для устройств 21CT..30CT")
    print("=" * 80)
    col_sair = [
        c for c in data.columns
        if "Sair" in c and any(c.startswith(f"{n}CT") for n in range(21, 31))
    ]
    print("col_Sair:", col_sair)
    print(data[col_sair].describe())

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=data[col_sair], ax=ax)
    ax.set_title("Boxplot параметра Sair Темп для устройств 21CT-30CT")
    ax.set_ylabel("Температура, °C")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("08_boxplot_sair.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("9. Ресемплирование по медиане (4 минуты), линейный график col_Sair")
    print("=" * 80)
    resampled = data[col_sair].resample("4min").median()
    fig, ax = plt.subplots(figsize=(16, 6))
    resampled.plot(ax=ax)
    ax.set_title("Параметр Sair Темп, ресемплировано по медиане (4 мин)")
    ax.set_xlabel("Время")
    ax.set_ylabel("Температура, °C")
    ax.legend(title="Устройство", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("09_resampled_sair.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("10. Сглаженный (rolling) график тренда для 2-3 признаков Sair")
    print("=" * 80)
    trend_cols = col_sair[:3]
    smoothed = data[trend_cols].rolling(window=200, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(16, 6))
    smoothed.plot(ax=ax)
    ax.set_title("Сглаженный тренд параметра Sair Темп (скользящее среднее, window=200)")
    ax.set_xlabel("Время")
    ax.set_ylabel("Температура, °C")
    ax.legend(title="Устройство", fontsize=8)
    plt.tight_layout()
    plt.savefig("10_rolling_trend.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("11. Сравнение: оригинал vs ресемплированный vs сглаженный")
    print("=" * 80)
    target_col = col_sair[0]
    # берём короткий интервал, чтобы разница между вариантами была видна
    window_start, window_end = data.index.min(), data.index.min() + pd.Timedelta(days=3)

    original_slice = data.loc[window_start:window_end, target_col]
    resampled_slice = data[target_col].resample("30min").median().loc[window_start:window_end]
    smoothed_slice = data[target_col].rolling(window=50, min_periods=1).mean().loc[window_start:window_end]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(original_slice.index, original_slice.values, label="Оригинал", alpha=0.5, linewidth=1)
    ax.plot(resampled_slice.index, resampled_slice.values, label="Ресемплировано (30 мин, медиана)", linewidth=2)
    ax.plot(smoothed_slice.index, smoothed_slice.values, label="Сглажено (rolling mean, window=50)", linewidth=2)
    ax.set_title(f"Сравнение предобработки признака: {target_col}")
    ax.set_xlabel("Время")
    ax.set_ylabel("Температура, °C")
    ax.legend()
    plt.tight_layout()
    plt.savefig("11_comparison.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("12. Гистограмма количества одновременно включённых устройств")
    print("=" * 80)
    # По условию: устройство включено, если признак состояния == 0.
    # Значения читаются как строки (среди них встречаются текстовые метки
    # 'Missing'/'Offln'), поэтому сравниваем по строковому представлению.
    ekc_state_on = data[state_cols].apply(
        lambda col: (col.astype(str).str.strip() == "0").astype(int)
    )
    devices_on_count = ekc_state_on.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    devices_on_count.value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("Количество одновременно включённых устройств")
    ax.set_xlabel("Число включённых устройств")
    ax.set_ylabel("Количество замеров (строк)")
    plt.tight_layout()
    plt.savefig("12_devices_on_hist.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("13. Гистограммы состояний для нескольких устройств")
    print("=" * 80)
    sample_state_cols = state_cols[:3]
    fig, axes = plt.subplots(1, len(sample_state_cols), figsize=(15, 5))
    for ax, col in zip(axes, sample_state_cols):
        sns.countplot(x=data[col].dropna(), ax=ax)
        ax.set_title(col.replace(": EKC состояние", ""))
        ax.set_xlabel("Код состояния")
        ax.set_ylabel("Количество замеров")
    fig.suptitle("Распределение устройств по состояниям (EKC состояние)")
    plt.tight_layout()
    plt.savefig("13_states_hist.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("14. Матрица корреляции признаков col_Sair")
    print("=" * 80)
    corr = data[col_sair].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Корреляция параметров Sair Темп (устройства 21CT-30CT)")
    plt.tight_layout()
    plt.savefig("14_correlation.png", dpi=120)
    plt.close()

    print("\n" + "=" * 80)
    print("15. Творческая гипотеза: температура u09 S5 vs состояние EKC")
    print("=" * 80)
    device_prefix = state_cols[0].split(":")[0]
    temp_col = f"{device_prefix}: u09 S5 Темп"
    state_col = state_cols[0]

    if temp_col in data.columns:
        stats_by_state = data.groupby(state_col)[temp_col].describe()
        print(stats_by_state)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=data[state_col].astype(str), y=data[temp_col], ax=ax)
        ax.set_title(f"Температура {temp_col} в зависимости от состояния устройства")
        ax.set_xlabel("Код состояния EKC")
        ax.set_ylabel("Температура, °C")
        plt.tight_layout()
        plt.savefig("15_hypothesis.png", dpi=120)
        plt.close()
        # Значения статуса читаются как строки, поэтому сделали так:
        on_key = next((idx for idx in stats_by_state.index if str(idx).strip() == "0"), None)
        off_means = stats_by_state.drop(index=on_key, errors="ignore")["mean"].dropna()
        on_mean = stats_by_state.loc[on_key, "mean"] if on_key is not None else None

        print("\nВывод:")
        print(f"  Средняя температура в состоянии 'включено' (0): {on_mean}")
        print(f"  Средняя температура в других состояниях:\n{off_means}")
        if on_mean is not None and len(off_means) and (off_means > on_mean).all():
            print("  Гипотеза подтверждается: в рабочем режиме температура ниже.")
        else:
            print("  Гипотеза требует уточнения на конкретных данных.")

    print("\nГотово. Графики сохранены в текущей папке (файлы 06_*.png ... 15_*.png).")


if __name__ == "__main__":
    main()